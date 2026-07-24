import hashlib
import json
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

from utils.p2t3 import (
    P2T3_DEEP_CONVERSATION_TYPE,
    P2T3_SHALLOW_CONVERSATION_TYPE,
    P2T3_SOURCE_TYPE,
    attach_p2t3_sequence_metadata,
)


def list_json_files(raw_dir, limit_files=None):
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            "P2T3 pre-training raw directory does not exist: {}".format(
                raw_dir
            )
        )
    files = sorted(raw_dir.glob("*.json"))
    if limit_files is not None:
        files = files[: max(0, int(limit_files))]
    if not files:
        raise FileNotFoundError(
            "No JSON files were found directly under {}.".format(raw_dir)
        )
    return files


def record_content(record):
    if "content" in record:
        return str(record["content"])
    if "word content" in record:
        return str(record["word content"])
    raise KeyError("P2T3 post record has neither 'content' nor 'word content'.")


def iter_post_texts(post):
    yield record_content(post["source"])
    comments = post.get("comment", [])
    if isinstance(comments, list):
        for comment in comments:
            yield record_content(comment)
        return

    for chain in comments.get("deep conversation", []):
        for comment in chain.get("comments", []):
            yield record_content(comment)
    for comment in comments.get("shallow conversation", []):
        yield record_content(comment)


class P2T3TextCorpus:
    """Re-iterable corpus for streaming Word2Vec construction."""

    def __init__(
        self,
        directories,
        language,
        tokenize_mode,
        limit_by_directory=None,
    ):
        self.directories = [Path(path) for path in directories]
        self.language = language
        self.tokenize_mode = tokenize_mode
        self.limit_by_directory = limit_by_directory or {}

    def __iter__(self):
        from utils.tools import word_tokenizer

        for directory in self.directories:
            limit = self.limit_by_directory.get(str(directory))
            files = list_json_files(directory, limit_files=limit)
            for path in files:
                with path.open("r", encoding="utf-8") as file_obj:
                    post = json.load(file_obj)
                for text in iter_post_texts(post):
                    tokens = word_tokenizer(
                        text,
                        lang=self.language,
                        mode=self.tokenize_mode,
                    )
                    if tokens:
                        yield tokens


def _metadata_data(x, node_ids, chain_ids, depths, type_ids, level_one_mask):
    data = Data(
        x=x.cpu(),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        directed_edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    data.p2t3_node_id = torch.tensor(node_ids, dtype=torch.long)
    data.p2t3_chain_id = torch.tensor(chain_ids, dtype=torch.long)
    data.p2t3_depth = torch.tensor(depths, dtype=torch.long)
    data.p2t3_type_id = torch.tensor(type_ids, dtype=torch.long)
    data.p2t3_level_one_mask = torch.tensor(
        level_one_mask,
        dtype=torch.bool,
    )
    data.p2t3_sequence_length = torch.tensor(
        [len(node_ids)],
        dtype=torch.long,
    )
    return data


def _chain_post_to_data(post, text_encoder, args):
    max_sequence_length = max(
        1,
        int(getattr(args, "p2t3_max_sequence_length", 1000)),
    )
    max_chain_length = max(
        1,
        int(getattr(args, "p2t3_max_chain_length", 40)),
    )
    max_chain_identifiers = max(
        1,
        int(
            getattr(
                args,
                "p2t3_max_chain_identifiers",
                getattr(args, "p2t3_d_model", 512),
            )
        ),
    )

    texts = [record_content(post["source"])]
    chain_ids = [0]
    depths = [0]
    type_ids = [P2T3_SOURCE_TYPE]
    level_one_mask = [False]
    conversation_id = 1

    comments = post.get("comment", {})
    conversation_groups = (
        (
            comments.get("deep conversation", []),
            P2T3_DEEP_CONVERSATION_TYPE,
        ),
        (
            comments.get("shallow conversation", []),
            P2T3_SHALLOW_CONVERSATION_TYPE,
        ),
    )
    for conversations, conversation_type in conversation_groups:
        for conversation in conversations:
            if (
                len(texts) >= max_sequence_length
                or conversation_id >= max_chain_identifiers
            ):
                break

            if conversation_type == P2T3_DEEP_CONVERSATION_TYPE:
                chain_comments = conversation.get("comments", [])
            else:
                chain_comments = [conversation]
            chain_comments = chain_comments[:max_chain_length]
            remaining = max_sequence_length - len(texts)
            chain_comments = chain_comments[:remaining]
            if not chain_comments:
                continue

            for offset, comment in enumerate(chain_comments):
                texts.append(record_content(comment))
                chain_ids.append(conversation_id)
                depths.append(int(comment.get("depth", offset + 1)))
                type_ids.append(conversation_type)
                level_one_mask.append(offset == 0)
            conversation_id += 1

    x = text_encoder.get_sentence_embeddings(texts)
    node_ids = list(range(len(texts)))
    return _metadata_data(
        x,
        node_ids,
        chain_ids,
        depths,
        type_ids,
        level_one_mask,
    )


def _flat_post_to_data(post, text_encoder, args):
    comments = post.get("comment", [])
    texts = [record_content(post["source"])]
    texts.extend(record_content(comment) for comment in comments)
    x = text_encoder.get_sentence_embeddings(texts).cpu()

    row = []
    col = []
    for comment in comments:
        parent = int(comment.get("parent", -1)) + 1
        child = int(comment.get("comment id", len(col))) + 1
        if 0 <= parent < len(texts) and 0 <= child < len(texts):
            row.append(parent)
            col.append(child)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        directed_edge_index=edge_index.clone(),
    )
    return attach_p2t3_sequence_metadata(data, args)


def p2t3_post_to_data(post, text_encoder, args):
    comments = post.get("comment", [])
    if isinstance(comments, dict):
        return _chain_post_to_data(post, text_encoder, args)
    if isinstance(comments, list):
        return _flat_post_to_data(post, text_encoder, args)
    raise TypeError("Unsupported P2T3 comment structure: {}".format(type(comments)))


def _encoder_fingerprint(text_encoder):
    path = getattr(text_encoder, "w2v_path", None)
    if path is None:
        return {
            "type": text_encoder.__class__.__name__,
            "embedding_dim": int(text_encoder.embedding_dim),
        }
    path = Path(path)
    stat = path.stat()
    return {
        "type": text_encoder.__class__.__name__,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "embedding_dim": int(text_encoder.embedding_dim),
    }


def pretrain_cache_signature(raw_dir, text_encoder, args, limit_files=None):
    payload = {
        "raw_dir": str(Path(raw_dir).resolve()),
        "encoder": _encoder_fingerprint(text_encoder),
        "max_sequence_length": int(
            getattr(args, "p2t3_max_sequence_length", 1000)
        ),
        "max_chain_length": int(
            getattr(args, "p2t3_max_chain_length", 40)
        ),
        "max_chain_identifiers": int(
            getattr(
                args,
                "p2t3_max_chain_identifiers",
                getattr(args, "p2t3_d_model", 512),
            )
        ),
        "limit_files": limit_files,
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()[:12]


def _safe_remove_generated_cache(cache_dir, processed_root):
    cache_dir = cache_dir.resolve()
    processed_root = processed_root.resolve()
    if (
        cache_dir.parent != processed_root
        or not cache_dir.name.startswith("ein_p2t3_")
    ):
        raise RuntimeError(
            "Refusing to remove unexpected cache directory: {}".format(
                cache_dir
            )
        )
    shutil.rmtree(cache_dir)


def prepare_p2t3_pretrain_cache(
    raw_dir,
    text_encoder,
    args,
    processed_root=None,
    chunk_size=512,
    force_rebuild=False,
    limit_files=None,
):
    raw_dir = Path(raw_dir)
    files = list_json_files(raw_dir, limit_files=limit_files)
    if processed_root is None:
        processed_root = raw_dir.parent / "processed"
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    signature = pretrain_cache_signature(
        raw_dir,
        text_encoder,
        args,
        limit_files=limit_files,
    )
    cache_dir = processed_root / "ein_p2t3_{}".format(signature)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file() and not force_rebuild:
        return cache_dir
    if cache_dir.exists():
        _safe_remove_generated_cache(cache_dir, processed_root)
    cache_dir.mkdir(parents=True)

    chunk_size = max(1, int(chunk_size))
    chunk = []
    chunks = []
    total = 0
    try:
        for index, path in enumerate(files, start=1):
            with path.open("r", encoding="utf-8") as file_obj:
                post = json.load(file_obj)
            chunk.append(p2t3_post_to_data(post, text_encoder, args))
            total += 1

            if len(chunk) >= chunk_size:
                filename = "chunk_{:06d}.pt".format(len(chunks))
                torch.save(chunk, cache_dir / filename)
                chunks.append({"file": filename, "count": len(chunk)})
                chunk = []

            if index % 1000 == 0 or index == len(files):
                print(
                    "P2T3 cache: processed {}/{} files".format(
                        index,
                        len(files),
                    ),
                    flush=True,
                )

        if chunk:
            filename = "chunk_{:06d}.pt".format(len(chunks))
            torch.save(chunk, cache_dir / filename)
            chunks.append({"file": filename, "count": len(chunk)})

        manifest = {
            "version": 1,
            "raw_dir": str(raw_dir.resolve()),
            "signature": signature,
            "total": total,
            "chunks": chunks,
        }
        with manifest_path.open("w", encoding="utf-8") as file_obj:
            json.dump(manifest, file_obj, indent=2, ensure_ascii=False)
    except Exception:
        if cache_dir.exists():
            _safe_remove_generated_cache(cache_dir, processed_root)
        raise
    return cache_dir


def _torch_load_data(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class P2T3PretrainChunkDataset(IterableDataset):
    def __init__(self, cache_dir, seed=0, shuffle=True):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        with (self.cache_dir / "manifest.json").open(
            "r",
            encoding="utf-8",
        ) as file_obj:
            self.manifest = json.load(file_obj)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __len__(self):
        return int(self.manifest["total"])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        chunks = list(self.manifest["chunks"])
        generator = random.Random(self.seed + self.epoch)
        if self.shuffle:
            generator.shuffle(chunks)
        chunks = chunks[worker_id::num_workers]

        for chunk_info in chunks:
            examples = _torch_load_data(
                self.cache_dir / chunk_info["file"]
            )
            if self.shuffle:
                generator.shuffle(examples)
            yield from examples
