import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch

from model.P2T3 import P2T3
from utils.p2t3_pretrain import (
    P2T3PretrainChunkDataset,
    p2t3_post_to_data,
    prepare_p2t3_pretrain_cache,
)


class FakeEncoder:
    embedding_dim = 5

    def get_sentence_embeddings(self, texts):
        rows = []
        for text in texts:
            value = float(len(text))
            rows.append(
                torch.tensor(
                    [value, value % 3, value % 5, 1.0, -1.0],
                    dtype=torch.float32,
                )
            )
        return torch.stack(rows)


def make_args():
    return SimpleNamespace(
        max_hop=4,
        p2t3_d_model=8,
        p2t3_num_layers=1,
        p2t3_num_heads=2,
        p2t3_dim_feedforward=16,
        p2t3_dropout=0.0,
        p2t3_max_sequence_length=16,
        p2t3_max_chain_length=8,
        p2t3_max_chain_identifiers=8,
        p2t3_max_depth=8,
        p2t3_measure="JSD",
        p2t3_unsup_weight=0.0,
    )


def chain_post(source_text):
    return {
        "source": {
            "content": source_text,
            "chain identifier": 0,
            "depth": 0,
            "type": 0,
        },
        "comment": {
            "deep conversation": [
                {
                    "chain identifier": 1,
                    "type": 1,
                    "comments": [
                        {"content": "deep one", "depth": 1},
                        {"content": "deep two", "depth": 2},
                    ],
                }
            ],
            "shallow conversation": [
                {
                    "content": "shallow",
                    "chain identifier": 2,
                    "depth": 1,
                    "type": 2,
                }
            ],
        },
    }


class P2T3PretrainDataTest(unittest.TestCase):
    def test_released_chain_json_becomes_cached_token_metadata(self):
        data = p2t3_post_to_data(
            chain_post("source"),
            FakeEncoder(),
            make_args(),
        )
        self.assertEqual(tuple(data.x.shape), (4, 5))
        self.assertEqual(data.p2t3_node_id.tolist(), [0, 1, 2, 3])
        self.assertEqual(data.p2t3_chain_id.tolist(), [0, 1, 1, 2])
        self.assertEqual(data.p2t3_depth.tolist(), [0, 1, 2, 1])
        self.assertEqual(data.p2t3_type_id.tolist(), [0, 1, 1, 2])
        self.assertEqual(
            data.p2t3_level_one_mask.tolist(),
            [False, True, False, True],
        )

    def test_flat_ein_json_is_also_supported(self):
        post = {
            "source": {"content": "source"},
            "comment": [
                {"comment id": 0, "parent": -1, "content": "reply"},
                {"comment id": 1, "parent": 0, "content": "nested"},
            ],
        }
        data = p2t3_post_to_data(post, FakeEncoder(), make_args())
        self.assertEqual(tuple(data.x.shape), (3, 5))
        self.assertEqual(data.p2t3_node_id.tolist(), [0, 1, 2])
        self.assertEqual(data.p2t3_type_id.tolist(), [0, 1, 1])

    def test_chunk_cache_feeds_native_mi_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "dataset" / "raw"
            processed_dir = root / "dataset" / "processed"
            raw_dir.mkdir(parents=True)
            for index in range(2):
                with (raw_dir / "{}.json".format(index)).open(
                    "w",
                    encoding="utf-8",
                ) as file_obj:
                    json.dump(chain_post("source {}".format(index)), file_obj)

            cache_dir = prepare_p2t3_pretrain_cache(
                raw_dir,
                FakeEncoder(),
                make_args(),
                processed_root=processed_dir,
                chunk_size=1,
            )
            dataset = P2T3PretrainChunkDataset(
                cache_dir,
                seed=0,
                shuffle=False,
            )
            examples = list(dataset)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(examples), 2)

            model = P2T3(
                in_feats=5,
                hid_feats=8,
                out_feats=8,
                num_classes=2,
                args=make_args(),
                device=torch.device("cpu"),
            )
            loss = model.pretraining_loss(Batch.from_data_list(examples))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(model.transformer_encoder.layers[0].linear1.weight.grad)


if __name__ == "__main__":
    unittest.main()
