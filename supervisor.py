import torch
from utils.tools import get_obj_from_str
import os
from utils.logger import get_result_name
from utils.word2vec import *
from utils.dataloader import *
from model.EIN_ResGCN import ResGCN
from model.EIN_ResGCN_Uncertainty import ResGCN_Uncertainty
from model.EIN_BiGCN import BiGCN
from model.EIN_BiGCN_Uncertainty import BiGCN_Uncertainty
from model.ResGCN_StateAuxSameDiff import ResGCN_StateAuxSameDiff
from model.BiGCN_StateAuxSameDiff import BiGCN_StateAuxSameDiff
from model.EIN_ResGCN_SameDiffFusion import EINResGCNSameDiffFusion
from model.EIN_BiGCN_SameDiffFusion import EINBiGCNSameDiffFusion
from model.BiGCN_UncertaintySemanticChange import BiGCN_UncertaintySemanticChange
from model.BiGCN_RevisionAwareSemanticChange import (
    BiGCN_RevisionAwareSemanticChange,
)
from model.ResGCN_UncertaintySemanticChange import ResGCN_UncertaintySemanticChange
from model.ResGCN_RevisionAwareSemanticChange import (
    ResGCN_RevisionAwareSemanticChange,
)
from model.GCN_UncertaintySemanticChange import GCN_UncertaintySemanticChange
from model.GIN_UncertaintySemanticChange import GIN_UncertaintySemanticChange
from model.KAGNN_UncertaintySemanticChange import KAGNN_UncertaintySemanticChange
from model.SEEGraphMAE import SEEGraphMAE
from model.KAGNN import KAGNN
from model.LIRS import LIRSGIN
from model.NEGT import NEGT
from model.RAGCL_baselines import RAGCLBiGCN, RAGCLResGCN
from trainer.EIN_trainer import EINTrainer
from trainer.LIRS_trainer import LIRSTrainer
from trainer.NEGT_trainer import NEGTTrainer
from trainer.RAGCL_trainer import RAGCLTrainer
from trainer.SEEGraphMAE_trainer import SEEGraphMAETrainer




def resolve_device(args):
    requested = str(getattr(args, 'device', 'cpu')).strip().lower()
    if requested.isdigit():
        requested = 'cuda:{}'.format(requested)
    elif requested.startswith('gpu') and requested[3:].isdigit():
        requested = 'cuda:{}'.format(requested[3:])
    elif requested.startswith('cuda') and requested[4:].isdigit():
        requested = 'cuda:{}'.format(requested[4:])

    if requested.startswith('cuda'):
        if not torch.cuda.is_available():
            device_nodes = [
                path
                for path in (
                    '/dev/nvidia0',
                    '/dev/nvidiactl',
                    '/dev/nvidia-uvm',
                )
                if os.path.exists(path)
            ]
            raise RuntimeError(
                'CUDA was requested ({}) but PyTorch cannot access a GPU. '
                'torch={} was built with CUDA {}, visible device count={}, '
                'mapped NVIDIA device nodes={}. This usually means the job or '
                'container was started without GPU device passthrough. Refusing '
                'to silently fall back to CPU.'.format(
                    requested,
                    torch.__version__,
                    torch.version.cuda,
                    torch.cuda.device_count(),
                    device_nodes,
                )
            )
        # torch.cuda.set_device() requires an explicit device index. A bare
        # "cuda" means the first device visible to this process.
        device = torch.device(
            'cuda:0' if requested == 'cuda' else requested
        )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                'Requested {}, but only {} CUDA device(s) are visible.'.format(
                    requested,
                    torch.cuda.device_count(),
                )
            )
        torch.cuda.set_device(device)
        allow_tf32 = bool(getattr(args, 'allow_tf32', True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_tf32:
            torch.set_float32_matmul_precision('high')
        return device

    if requested != 'cpu':
        raise ValueError('Unsupported device: {}'.format(requested))
    return torch.device('cpu')


def build_text_encoder(args, device, label_source_path):
    if args.word_embedding == 'word2vec':
        model_path = os.path.join('word2vec',
                            f'w2v_{args.dataset}_{args.tokenize_mode}_{args.vector_size}.model')

        if not os.path.exists(model_path):
            sentences = collect_sentences(label_source_path, args.language, args.tokenize_mode)
            w2v_model = train_word2vec(sentences, args.vector_size, args.seed)
            w2v_model.save(model_path)

        encoder = Embedding(model_path, args.language, args.tokenize_mode)
    elif args.word_embedding == 'multilingual-e5-base':
        encoder = MultilingualE5Embedding(
            model_name=getattr(args, 'e5_model_name', 'intfloat/multilingual-e5-base'),
            device=device,
            max_length=getattr(args, 'e5_max_length', 128),
            batch_size=getattr(args, 'e5_batch_size', 64),
            local_files_only=getattr(args, 'e5_local_files_only', False)
        )
        args.in_feats = encoder.embedding_dim
    else:
        raise ValueError('Unsupported word_embedding: {}'.format(args.word_embedding))

    return encoder


def parse_ood_source_datasets(args):
    source_datasets = getattr(args, 'ood_source_datasets', [])
    if source_datasets is None:
        source_datasets = []
    if isinstance(source_datasets, str):
        source_datasets = [dataset.strip() for dataset in source_datasets.split(',') if dataset.strip()]
    return source_datasets


def get_dataset_cache_name(args):
    result_name = get_result_name(args)
    if result_name:
        return result_name
    return 'pid_{}'.format(os.getpid())


def dataset_paths(args, dataset):
    label_source_path = os.path.join('dataset', dataset, 'source')
    label_dataset_path = os.path.join(
        'dataset',
        dataset,
        'dataset_cache',
        get_dataset_cache_name(args),
        'seed_{}'.format(args.seed)
    )
    return label_source_path, label_dataset_path


def split_manifest_path(args, dataset):
    filename = 'split_{}_k{}_seed{}.json'.format(
        args.split,
        args.k,
        args.seed,
    )
    return os.path.join('dataset', dataset, 'splits', filename)


def split_and_get_paths(args, dataset):
    label_source_path, label_dataset_path = dataset_paths(args, dataset)
    split_dataset(
        label_source_path,
        label_dataset_path,
        k_shot=args.k,
        split=args.split,
        seed=args.seed,
        split_manifest_path=split_manifest_path(args, dataset),
    )
    train_path = os.path.join(label_dataset_path, 'train')
    val_path = os.path.join(label_dataset_path, 'val')
    test_path = os.path.join(label_dataset_path, 'test')
    return train_path, val_path, test_path


def build_id_paths(args):
    label_source_path, label_dataset_path = dataset_paths(args, args.dataset)
    train_post, val_post, test_post = build_split_posts(
        label_source_path,
        args.k,
        args.split,
        seed=args.seed,
        split_manifest_path=split_manifest_path(args, args.dataset),
    )
    return write_split_posts(label_dataset_path, train_post, val_post, test_post)


def build_strict_ood_paths(args):
    target_source_path, target_dataset_path = dataset_paths(args, args.dataset)
    target_train_post, target_val_post, target_test_post = build_split_posts(
        target_source_path,
        args.k,
        args.split,
        seed=args.seed,
        split_manifest_path=split_manifest_path(args, args.dataset),
    )

    source_datasets = parse_ood_source_datasets(args)
    if len(source_datasets) < 2:
        raise ValueError('Strict OOD expects at least two source datasets in ood_source_datasets.')
    if args.dataset in source_datasets:
        raise ValueError('Strict OOD source datasets must not include target dataset {}.'.format(args.dataset))

    source_train_post = []
    source_val_post = []
    for domain_id, source_dataset in enumerate(source_datasets):
        source_path = os.path.join('dataset', source_dataset, 'source')
        train_post, val_post, _ = build_split_posts(
            source_path,
            args.k,
            args.split,
            seed=args.seed,
            split_manifest_path=split_manifest_path(args, source_dataset),
        )
        source_train_post.extend(assign_domain_id(train_post, domain_id))
        source_val_post.extend(assign_domain_id(val_post, domain_id))

    train_post = strict_balanced_sample_posts(source_train_post, target_train_post)
    val_domain = getattr(args, 'ood_val_domain', 'source')
    if val_domain == 'source':
        val_post = strict_balanced_sample_posts(source_val_post, target_val_post)
    elif val_domain == 'target':
        val_post = target_val_post
    else:
        raise ValueError('Unsupported ood_val_domain: {}'.format(val_domain))

    print('Strict OOD target train counts: {}'.format(post_class_counts(target_train_post)))
    print('Strict OOD sampled train counts: {}'.format(post_class_counts(train_post)))
    print('Strict OOD target val counts: {}'.format(post_class_counts(target_val_post)))
    print('Strict OOD selected val counts: {}'.format(post_class_counts(val_post)))
    print('Strict OOD target test counts: {}'.format(post_class_counts(target_test_post)))

    return write_split_posts(target_dataset_path, train_post, val_post, target_test_post)


def load_graph_dataset(args, path, text_encoder):
    if args.base_model in [
        'ResGCN',
        'ResGCN_Uncertainty',
        'ResGCN_StateAuxSameDiff',
        'ResGCN_SameDiffFusion',
        'ResGCN_UncertaintySemanticChange',
        'GCN_UncertaintySemanticChange',
        'GIN_UncertaintySemanticChange',
        'KAGNN_UncertaintySemanticChange',
        'SEEGraphMAE',
        'KAGNN',
        'NEGT',
    ]:
        return ResGCNTreeDataset(path, args.word_embedding, text_encoder, args.undirected, args=args)
    if args.base_model in [
        'BiGCN',
        'BiGCN_Uncertainty',
        'BiGCN_StateAuxSameDiff',
        'BiGCN_SameDiffFusion',
        'BiGCN_UncertaintySemanticChange',
        'LIRS',
    ]:
        return TreeDataset(path, args.word_embedding, text_encoder, args=args)
    if args.base_model in ['RAGCL_ResGCN', 'RAGCL_BiGCN', 'Plain_ResGCN', 'Plain_BiGCN']:
        return ResGCNTreeDataset(path, args.word_embedding, text_encoder, args.undirected, args=args)
    raise ValueError('Unsupported base_model: {}'.format(args.base_model))


def build_experiment_datasets(args, text_encoder):
    experiment_mode = getattr(args, 'experiment_mode', 'id')
    if experiment_mode == 'id':
        train_path, val_path, test_path = build_id_paths(args)
        return (
            load_graph_dataset(args, train_path, text_encoder),
            load_graph_dataset(args, val_path, text_encoder),
            load_graph_dataset(args, test_path, text_encoder)
        )

    if experiment_mode != 'strict_ood':
        raise ValueError('Unsupported experiment_mode: {}'.format(experiment_mode))

    if args.word_embedding == 'word2vec':
        raise ValueError('Strict OOD requires a shared encoder. Use multilingual-e5-base for now.')

    train_path, val_path, test_path = build_strict_ood_paths(args)
    return (
        load_graph_dataset(args, train_path, text_encoder),
        load_graph_dataset(args, val_path, text_encoder),
        load_graph_dataset(args, test_path, text_encoder)
    )


def EIN_ResGCN_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)
    
    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)
    
    print('Seed {} | Initializing ResGCN'.format(args.seed), flush=True)
    base_model =  ResGCN(dataset=train_dataset, num_classes=args.num_classes, hidden=args.hidden_dim,
                            num_feat_layers=args.n_layers_feat, num_conv_layers=args.n_layers_conv,
                            num_fc_layers=args.n_layers_fc, gfn=False, collapse=False,
                            residual=args.skip_connection,
                            res_branch=args.res_branch, global_pool=args.global_pool, dropout=args.dropout,
                            edge_norm=args.edge_norm, args=args, device=device).to(device)

    optimizer = base_model.init_optimizer(args)

    datasets = [train_dataset, val_dataset, test_dataset]

    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_ResGCN_Uncertainty_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing ResGCN_Uncertainty'.format(args.seed), flush=True)
    base_model = ResGCN_Uncertainty(
        dataset=train_dataset,
        num_classes=args.num_classes,
        hidden=args.hidden_dim,
        num_feat_layers=args.n_layers_feat,
        num_conv_layers=args.n_layers_conv,
        num_fc_layers=args.n_layers_fc,
        gfn=False,
        collapse=False,
        residual=args.skip_connection,
        res_branch=args.res_branch,
        global_pool=args.global_pool,
        dropout=args.dropout,
        edge_norm=args.edge_norm,
        args=args,
        device=device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()



def EIN_BiGCN_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)
    
    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing BiGCN'.format(args.seed), flush=True)
    base_model = BiGCN(args.in_feats, args.hidden_dim, args.hidden_dim, args.num_classes, args, device).to(device)


    optimizer = base_model.init_optimizer(args)

    datasets = [train_dataset, val_dataset, test_dataset]

    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_BiGCN_Uncertainty_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing BiGCN_Uncertainty'.format(args.seed), flush=True)
    base_model = BiGCN_Uncertainty(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_RAGCL_ResGCN_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing RAGCL ResGCN'.format(args.seed), flush=True)
    base_model = RAGCLResGCN(
        dataset=train_dataset,
        num_classes=args.num_classes,
        hidden=args.hidden_dim,
        num_feat_layers=args.n_layers_feat,
        num_conv_layers=args.n_layers_conv,
        num_fc_layers=args.n_layers_fc,
        gfn=False,
        collapse=False,
        residual=args.skip_connection,
        res_branch=args.res_branch,
        global_pool=args.global_pool,
        dropout=args.dropout,
        edge_norm=args.edge_norm,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = RAGCLTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_RAGCL_BiGCN_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing RAGCL BiGCN'.format(args.seed), flush=True)
    base_model = RAGCLBiGCN(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        getattr(args, 'TDdroprate', 0.0),
        getattr(args, 'BUdroprate', 0.0),
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = RAGCLTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_Plain_ResGCN_supervisor(args):
    args.use_unsup_loss = False
    return EIN_RAGCL_ResGCN_supervisor(args)


def EIN_Plain_BiGCN_supervisor(args):
    args.use_unsup_loss = False
    return EIN_RAGCL_BiGCN_supervisor(args)


def EIN_NEGT_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing NEGT'.format(args.seed), flush=True)
    base_model = NEGT(
        args.in_feats,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = NEGTTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_BiGCN_StateAuxSameDiff_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing BiGCN_StateAuxSameDiff'.format(args.seed), flush=True)
    base_model = BiGCN_StateAuxSameDiff(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_ResGCN_StateAuxSameDiff_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing ResGCN_StateAuxSameDiff'.format(args.seed), flush=True)
    base_model = ResGCN_StateAuxSameDiff(
        dataset=train_dataset,
        num_classes=args.num_classes,
        hidden=args.hidden_dim,
        num_feat_layers=args.n_layers_feat,
        num_conv_layers=args.n_layers_conv,
        num_fc_layers=args.n_layers_fc,
        gfn=False,
        collapse=False,
        residual=args.skip_connection,
        res_branch=args.res_branch,
        global_pool=args.global_pool,
        dropout=args.dropout,
        edge_norm=args.edge_norm,
        args=args,
        device=device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_ResGCN_SameDiffFusion_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing ResGCN_SameDiffFusion'.format(args.seed), flush=True)
    
    #这里的base_model是模型的实例化对象，里面包含了模型的forward方法和loss计算方法的定义以及各个模型的具体实现
    base_model = EINResGCNSameDiffFusion(
        dataset=train_dataset,
        num_classes=args.num_classes,
        hidden=args.hidden_dim,
        num_feat_layers=args.n_layers_feat,
        num_conv_layers=args.n_layers_conv,
        num_fc_layers=args.n_layers_fc,
        gfn=False,
        collapse=False,
        residual=args.skip_connection,
        res_branch=args.res_branch,
        global_pool=args.global_pool,
        dropout=args.dropout,
        edge_norm=args.edge_norm,
        args=args,
        device=device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]

    #这里的Trainer类是整个训练验证测试的总控制器，里面会调用base_model参数的forward方法和loss计算方法
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_BiGCN_SameDiffFusion_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing BiGCN_SameDiffFusion'.format(args.seed), flush=True)
    base_model = EINBiGCNSameDiffFusion(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_BiGCN_UncertaintySemanticChange_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing BiGCN_UncertaintySemanticChange'.format(
            args.seed
        ),
        flush=True,
    )
    base_model = BiGCN_UncertaintySemanticChange(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_BiGCN_RevisionAwareSemanticChange_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing BiGCN_RevisionAwareSemanticChange'.format(
            args.seed
        ),
        flush=True,
    )
    base_model = BiGCN_RevisionAwareSemanticChange(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_ResGCN_UncertaintySemanticChange_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing ResGCN_UncertaintySemanticChange'.format(
            args.seed
        ),
        flush=True,
    )
    base_model = ResGCN_UncertaintySemanticChange(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_ResGCN_RevisionAwareSemanticChange_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing ResGCN_RevisionAwareSemanticChange'.format(
            args.seed
        ),
        flush=True,
    )
    base_model = ResGCN_RevisionAwareSemanticChange(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_GCN_UncertaintySemanticChange_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing GCN_UncertaintySemanticChange'.format(
            args.seed
        ),
        flush=True,
    )
    base_model = GCN_UncertaintySemanticChange(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_GIN_UncertaintySemanticChange_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing GIN_UncertaintySemanticChange'.format(
            args.seed
        ),
        flush=True,
    )
    base_model = GIN_UncertaintySemanticChange(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_KAGNN_UncertaintySemanticChange_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing KAGNN_UncertaintySemanticChange ({})'.format(
            args.seed,
            getattr(args, 'kagnn_variant', 'KAGCN'),
        ),
        flush=True,
    )
    base_model = KAGNN_UncertaintySemanticChange(
        args.in_feats,
        args.hidden_dim,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_SEEGraphMAE_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing SEEGraphMAE'.format(args.seed),
        flush=True,
    )
    base_model = SEEGraphMAE(
        args.in_feats,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = SEEGraphMAETrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_KAGNN_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print(
        'Seed {} | Initializing KAGNN ({})'.format(
            args.seed,
            getattr(args, 'kagnn_variant', 'KAGCN'),
        ),
        flush=True,
    )
    base_model = KAGNN(
        args.in_feats,
        args.hidden_dim,
        args.num_classes,
        args,
        device,
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = EINTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()


def EIN_LIRS_supervisor(args):
    init_seed(args.seed, need_deepfix=True)

    device = resolve_device(args)

    label_source_path, _ = dataset_paths(args, args.dataset)
    print('Seed {} | Building text encoder on {}'.format(args.seed, device), flush=True)
    text_encoder = build_text_encoder(args, device, label_source_path)

    print('Seed {} | Building experiment datasets'.format(args.seed), flush=True)
    train_dataset, val_dataset, test_dataset = build_experiment_datasets(args, text_encoder)

    print('Seed {} | Initializing LIRS'.format(args.seed), flush=True)
    base_model = LIRSGIN(
        args.in_feats,
        args.hidden_dim,
        args.num_classes,
        args,
        device
    ).to(device)

    optimizer = base_model.init_optimizer(args)
    datasets = [train_dataset, val_dataset, test_dataset]
    trainer = LIRSTrainer(datasets, base_model, optimizer, args, device)

    print('Seed {} | Start training'.format(args.seed), flush=True)
    return trainer.train_process()
