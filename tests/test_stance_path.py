"""Behavioral checks for trajectory isolation, stochastic compression and training."""

import ast
from copy import deepcopy
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
import yaml

from model.StancePath import StancePath, prepare_reply_tree, compute_stance_path_loss
from trainer.StancePath_trainer import StancePathTrainer, load_stance_path_checkpoint


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_torch_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph(label=0):
    # Two leaves share node 1; node 4 is a separate first-level branch.
    return SimpleNamespace(
        x=torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0],
                        [0, 0, 0, 1.], [1., 1., 0, 0]]),
        directed_edge_index=torch.tensor([[0, 1, 1, 0], [1, 2, 3, 4]]),
        directed_edge_stance=torch.tensor([0, 1, 1, 1]),
        y=torch.tensor([label]),
    )


def model(**values):
    torch.manual_seed(31)
    args = dict(dropout=0.0, stance_path_heads=2)
    args.update(values)
    return StancePath(4, 8, 2, SimpleNamespace(**args))


def config(dataset="Pheme", **updates):
    values = yaml.safe_load((ROOT / "configs" / "EIN" / f"{dataset}_StancePath_word2vec.yaml").read_text())
    values.update(updates)
    return SimpleNamespace(**values)


def test_original_paths_keep_local_stance_and_shared_prefix_once():
    raw = graph()
    # A symmetrized edge_index must not override the original reply relation.
    raw.edge_index = torch.cat((raw.directed_edge_index, raw.directed_edge_index.flip(0)), 1)
    tree = prepare_reply_tree(raw)
    assert {tuple(path.tolist()) for path in tree.paths} == {(1, 2), (1, 3), (4,)}
    assert tree.relation.tolist() == [2, 0, 1, 1, 1]
    net = model().eval()
    _, details = net(tree, full_paths=True, return_details=True)
    assert details[0]["selected_node_ids"].tolist() == [0, 1, 2, 3, 4]
    assert details[0]["node_weights"].tolist() == [1.] * 5
    # Repeated denials remain two parent-relative denials, never a root support.
    raw.directed_edge_stance[0] = 1
    assert prepare_reply_tree(raw).relation.tolist() == [2, 1, 1, 1, 1]


def test_discarded_branch_cannot_reach_classifier_even_in_multiple_layers():
    net = model(stance_path_interaction_layers=2).eval()
    raw = graph()
    tree = prepare_reply_tree(raw)
    mask = torch.tensor([float(2 in path.tolist()) for path in tree.paths])
    before, details = net(raw, path_masks=[mask], return_details=True)
    assert details[0]["selected_node_ids"].tolist() == [0, 1, 2]
    altered = deepcopy(raw)
    altered.x[[3, 4]] = torch.tensor([1e5, -2e5, 3e5, -4e5])
    altered.directed_edge_stance[[2, 3]] = 0
    torch.testing.assert_close(before, net(altered, path_masks=[mask]), atol=0, rtol=0)


def test_empty_selection_is_legal_and_source_prediction_has_no_reply_leakage():
    net = model(stance_path_threshold=1.0).eval()
    raw = graph()
    before, details = net(raw, return_details=True)
    assert details[0]["selected_node_ids"].tolist() == [0]
    assert details[0]["path_mask"].sum() == 0
    raw.x[1:] = 9999
    torch.testing.assert_close(before, net(raw), atol=0, rtol=0)
    singleton = SimpleNamespace(x=raw.x[:1], y=None)
    torch.testing.assert_close(before, net(singleton), atol=1e-7, rtol=1e-6)


def test_node_relabeling_and_batching_do_not_change_predictions():
    net = model().eval()
    raw = graph()
    permutation = torch.tensor([3, 4, 1, 0, 2])
    inverse = permutation.argsort()
    renamed = deepcopy(raw)
    renamed.x = raw.x[permutation]
    renamed.directed_edge_index = inverse[raw.directed_edge_index]
    renamed.root_index = inverse[torch.tensor([0])]
    torch.testing.assert_close(net(raw), net(renamed), atol=1e-6, rtol=1e-5)
    singleton = SimpleNamespace(x=torch.ones(1, 4))
    together = net([raw, singleton])
    torch.testing.assert_close(together, torch.cat((net(raw), net(singleton))))
    raw.y = None
    torch.testing.assert_close(together[:1], net(raw))


def test_pyg_batch_round_trip_when_available():
    pyg = pytest.importorskip("torch_geometric.data")
    data = graph()
    graphs = [pyg.Data(**vars(data)), pyg.Data(**vars(deepcopy(data)))]
    graphs[1].x = graphs[1].x + 0.4
    net = model().eval()
    torch.testing.assert_close(net(pyg.Batch.from_data_list(graphs)), net(graphs))


def test_concrete_sampling_is_random_and_classification_trains_the_selector():
    net = model().train()
    data = graph()
    data.x.requires_grad_(True)
    output, first = net(data, return_details=True)
    _, second = net(data, return_details=True)
    assert not torch.equal(first[0]["path_mask"], second[0]["path_mask"])
    assert ((first[0]["path_mask"] > 0) & (first[0]["path_mask"] < 1)).all()
    F.cross_entropy(output, data.y).backward()
    assert net.selector[-1].weight.grad.abs().sum() > 0
    assert net.field[0].weight.grad.abs().sum() > 0
    assert data.x.grad is None  # fixed text embeddings


def test_sufficiency_is_per_event_hinge_and_teacher_receives_no_gradient():
    logits = torch.tensor([[0., 0.], [-2., 2.]], requires_grad=True)
    teacher = torch.tensor([[2., -2.], [-2., 2.]], requires_grad=True)
    target = torch.tensor([0, 1])
    details = [dict(mask_kl=torch.tensor(0.2)), dict(mask_kl=torch.tensor(0.4))]
    loss, terms = compute_stance_path_loss(logits, target, details, teacher,
                                          sufficiency_weight=2., sufficiency_margin=0.1, kl_weight=0.3)
    ce = F.cross_entropy(logits, target, reduction="none")
    expected_suff = (ce - F.cross_entropy(teacher.detach(), target, reduction="none") - .1).relu()
    torch.testing.assert_close(terms["sufficiency"], expected_suff.mean())
    torch.testing.assert_close(loss, ce.mean() + 2 * expected_suff.mean() + .09)
    loss.backward()
    assert logits.grad.abs().sum() > 0
    assert teacher.grad is None
    with pytest.raises(ValueError, match="teacher"):
        compute_stance_path_loss(logits, target, details)


def test_mask_kl_is_bernoulli_sum_and_zero_at_prior():
    net = model(stance_path_prior=0.2).eval()
    _, details = net(graph(), return_details=True)
    q = details[0]["probabilities"]
    expected = (q * (q / .2).log() + (1 - q) * ((1 - q) / .8).log()).sum()
    torch.testing.assert_close(details[0]["mask_kl"], expected)
    with torch.no_grad():
        net.selector[-1].bias.fill_(torch.logit(torch.tensor(.2)))
    _, details = net(graph(), return_details=True)
    assert abs(details[0]["mask_kl"].item()) < 1e-6


@pytest.mark.parametrize("failure", ["missing_stance", "unknown_stance", "two_parents", "disconnected", "cycle"])
def test_invalid_reply_data_is_rejected_without_synthetic_repairs(failure):
    data = graph()
    if failure == "missing_stance":
        del data.directed_edge_stance
    elif failure == "unknown_stance":
        data.directed_edge_stance[0] = -1
    elif failure == "two_parents":
        data.directed_edge_index[1, 2] = 2
    elif failure == "disconnected":
        data.directed_edge_index = data.directed_edge_index[:, :-1]
        data.directed_edge_stance = data.directed_edge_stance[:-1]
    else:
        data.directed_edge_index[0, 0] = 2  # unreachable 1 <-> 2 cycle
    with pytest.raises(ValueError):
        prepare_reply_tree(data)


def test_without_stance_ablation_does_not_require_or_use_labels():
    net = model(stance_path_use_stance=False).eval()
    raw = graph()
    before = net(raw)
    del raw.directed_edge_stance
    torch.testing.assert_close(before, net(raw))


def load_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("dataset", ["Pheme", "Weibo", "DRWeibo"])
def test_configs_use_existing_stance_cache_and_main_dispatch(dataset):
    args = config(dataset)
    namespace = load_functions("supervisor.py", ["_as_bool", "_safe_cache_part", "_graph_dataset_cache_part",
                               "_requires_ragcl_centrality", "get_dataset_cache_name", "load_graph_dataset"],
                               {"re": re, "ResGCNTreeDataset": Mock()})
    baseline = SimpleNamespace(**yaml.safe_load(
        (ROOT / "configs" / "EIN" / f"{dataset}_SHPA_word2vec.yaml").read_text()))
    assert namespace["get_dataset_cache_name"](args) == namespace["get_dataset_cache_name"](baseline)
    namespace["load_graph_dataset"](args, "unused", None)
    namespace["ResGCNTreeDataset"].assert_called_once_with("unused", "word2vec", None, False, args=args)
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.ImportFrom) and node.module == "supervisor"
               and any(alias.name == "EIN_StancePath_supervisor" for alias in node.names) for node in tree.body)
    StancePath(args.in_feats, args.hidden_dim, args.num_classes, args)


def test_supervisor_reuses_ready_cache_without_loading_text_encoder():
    args = config(in_feats=4, hidden_dim=8, device="cpu")
    datasets = ([graph()], [graph()], [graph()])
    trainer_factory = Mock()
    trainer_factory.return_value.train_process.return_value = {"acc": 1.0}
    encoder = Mock(side_effect=AssertionError("Cached features must not load the encoder"))
    namespace = load_functions("supervisor.py", ["EIN_StancePath_supervisor"], {
        "init_seed": Mock(), "resolve_device": lambda _: torch.device("cpu"),
        "dataset_paths": lambda *_: ("source", "cache"),
        "load_cached_experiment_datasets": Mock(return_value=datasets),
        "build_text_encoder": encoder, "StancePath": StancePath, "StancePathTrainer": trainer_factory,
    })
    assert namespace["EIN_StancePath_supervisor"](args) == {"acc": 1.0}
    assert trainer_factory.call_args.args[0] is datasets
    encoder.assert_not_called()


def test_two_stage_training_freezes_teacher_and_checkpoint_is_student_only(tmp_path, monkeypatch):
    import trainer.StancePath_trainer as implementation
    monkeypatch.setattr(implementation, "get_log_dir", lambda _: str(tmp_path))
    args = config(in_feats=4, hidden_dim=8, dropout=0., batch_size=2, device="cpu",
                  stance_path_teacher_epochs=1, stance_path_student_epochs=2,
                  stance_path_student_patience=2, stance_path_threshold=.6)
    datasets = tuple([graph(0), graph(1)] for _ in range(3))

    class AuditedTrainer(StancePathTrainer):
        def _train_stage(self, net, stage, path, teacher=None):
            if stage == "student":
                assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
                frozen = deepcopy(teacher.state_dict())
                for key, value in net.state_dict().items():
                    torch.testing.assert_close(value, frozen[key])
            super()._train_stage(net, stage, path, teacher)
            if stage == "student":
                for key, value in teacher.state_dict().items():
                    torch.testing.assert_close(value, frozen[key], atol=0, rtol=0)

    trainer = AuditedTrainer(datasets, StancePath(4, 8, 2, args), args, "cpu")
    original_evaluate = trainer.evaluate
    tested = []

    def evaluate(net, loader, **kwargs):
        if loader is trainer.loaders["test"]:
            assert {row["stage"] for row in trainer.history} == {"teacher", "student"}
            tested.append(True)
        return original_evaluate(net, loader, **kwargs)

    trainer.evaluate = evaluate
    result = trainer.train_process()
    assert tested and {"acc", "auc", "f1", "retained_path_ratio", "source_only_ratio"} <= result.keys()
    restored, checkpoint = load_stance_path_checkpoint(trainer.best_path)
    assert checkpoint["stage"] == "student"
    assert restored.threshold == .6
    assert not any(key.startswith("teacher.") for key in checkpoint["model_state_dict"])
    torch.testing.assert_close(restored(graph()), trainer.model(graph()))
    assert (tmp_path / "test_results.json").is_file()
    # Evaluation does not need the teacher checkpoint or training/validation data.
    trainer.teacher_path.unlink()
    args.eval_only, args.checkpoint_path = True, str(trainer.best_path)
    evaluation = StancePathTrainer((None, None, datasets[2]), model(), args, "cpu")
    assert evaluation.teacher is None and evaluation.model.threshold == .6
    assert evaluation.train_process()["acc"] == result["acc"]
