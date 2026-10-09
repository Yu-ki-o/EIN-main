"""Reproducible, source-selected zero-shot OOD splits (standard library only).

Source and target records are deduplicated before splitting. Target labels never
control source membership, class balance, sample count, or validation selection.
Original post JSON files are read only; manifests bind their complete SHA-256.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

VERSION = 1
DATASETS = ("DRWeibo", "Weibo", "Pheme")
PROTOCOLS = ("cross_dataset", "drweibo_theme", "pheme_event")
_UNKNOWN_THEMES = {"", "none", "null", "nan", "unknown", "未知", "未分类"}


def _canonical_dataset(value):
    for name in DATASETS:
        if str(value).lower() == name.lower():
            return name
    raise ValueError("Unsupported OOD dataset: {!r}; choose {}".format(value, DATASETS))


def normalize_root_text(text):
    """Exact content deduplication after case and whitespace normalization."""
    return re.sub(r"\s+", "", str(text or "")).casefold()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_digest(manifest):
    """Hash every persisted field except the self-referential fingerprint."""
    return hashlib.sha256(_json_bytes({k: v for k, v in manifest.items() if k != "fingerprint"})).hexdigest()


def _read_post(path):
    raw = Path(path).read_bytes()
    post = json.loads(raw)
    source = post.get("source", {})
    if source.get("tweet id") is None or not str(source.get("tweet id", "")).strip():
        raise ValueError("Missing root tweet id: {}".format(path))
    if "content" not in source or source.get("label") not in (0, 1):
        raise ValueError("Expected root content and binary label 0/1: {}".format(path))
    if not isinstance(post.get("comment"), list):
        raise ValueError("Expected comment list: {}".format(path))
    return post, hashlib.sha256(raw).hexdigest()


def _record_path(manifest, record):
    dataset = record["dataset"]
    if dataset not in manifest["source_roots"]:
        raise ValueError("No source root for dataset {}".format(dataset))
    root = Path(manifest["source_roots"][dataset]).expanduser().resolve()
    relative = Path(record["file"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Manifest file escapes source directory: {}".format(relative))
    if relative.suffix.lower() != ".json":
        raise ValueError("Expected a JSON post file: {}".format(relative))
    return path


def _duplicate_keys(dataset, post):
    platform = "weibo" if dataset in ("DRWeibo", "Weibo") else "twitter"
    keys = [("id", platform, str(post["source"]["tweet id"]))]
    text = normalize_root_text(post["source"]["content"])
    if text:
        keys.append(("text", text))
    return keys


def read_pheme_events(archive_path):
    """Read thread IDs from archive paths; auto-detect mislabeled compression.

    No archive member is extracted or executed. The local archive commonly has
    a .tar.bz2 suffix despite containing gzip data, so use tarfile's r:* mode.
    """
    mapping = {}
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if any(part.startswith("._") for part in parts):
                continue
            for index, part in enumerate(parts):
                suffix = "-all-rnr-threads"
                if not part.endswith(suffix) or index + 2 >= len(parts):
                    continue
                if parts[index + 1] not in ("rumours", "non-rumours"):
                    continue
                root_id = parts[index + 2]
                if not root_id.isdigit():
                    continue
                event = part[:-len(suffix)]
                previous = mapping.setdefault(root_id, event)
                if previous != event:
                    raise ValueError("PHEME root {} belongs to multiple events".format(root_id))
    if not mapping:
        raise ValueError("No PHEME event/thread paths found in {}".format(archive_path))
    return mapping


def _load_dataset(dataset, root, workers):
    directory = (Path(root).expanduser().resolve() / dataset / "source")
    if not directory.is_dir():
        raise ValueError("Missing source directory: {}".format(directory))
    files = sorted(directory.glob("*.json"))
    if not files:
        raise ValueError("No JSON posts in {}".format(directory))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(_read_post, files))
    return directory, [
        {"dataset": dataset, "file": path.name, "sha256": digest,
         "domain": dataset, "post": post}
        for path, (post, digest) in zip(files, loaded)
    ]


def _sort_key(record):
    return record["dataset"], record["file"]


def _duplicate_components(records):
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owner = {}
    for i, record in enumerate(records):
        for key in _duplicate_keys(record["dataset"], record["post"]):
            if key in owner:
                parents[find(i)] = find(owner[key])
            else:
                owner[key] = i
    components = defaultdict(list)
    for i, record in enumerate(records):
        components[find(i)].append(record)
    return [sorted(values, key=_sort_key) for values in components.values()]


def _deduplicate(source, target):
    """Retain a target-independent source pool and remove test contamination.

    A source dataset always has the same train/validation membership when tested
    on different target datasets. All original source IDs/text are excluded from
    the target, including source components dropped for conflicting labels.
    """
    result_source, result_target = [], []
    counts = Counter({
        "source_removed_conflicting_labels": 0,
        "source_removed_duplicates": 0,
        "target_removed_source_overlap": 0,
        "target_removed_conflicting_labels": 0,
        "target_removed_duplicates": 0,
    })
    source_keys = {key for r in source for key in _duplicate_keys(r["dataset"], r["post"])}
    for component in _duplicate_components(source):
        if len({r["post"]["source"]["label"] for r in component}) > 1:
            counts["source_removed_conflicting_labels"] += len(component)
        else:
            result_source.append(component[0])
            counts["source_removed_duplicates"] += len(component) - 1
    for component in _duplicate_components(target):
        overlaps = any(key in source_keys for r in component for key in _duplicate_keys(r["dataset"], r["post"]))
        if overlaps:
            counts["target_removed_source_overlap"] += len(component)
        elif len({r["post"]["source"]["label"] for r in component}) > 1:
            counts["target_removed_conflicting_labels"] += len(component)
        else:
            result_target.append(component[0])
            counts["target_removed_duplicates"] += len(component) - 1
    return sorted(result_source, key=_sort_key), sorted(result_target, key=_sort_key), dict(counts)


def _source_split(records, seed, val_ratio):
    buckets = defaultdict(list)
    for record in records:
        buckets[(record["domain"], int(record["post"]["source"]["label"]))].append(record)
    train, val = [], []
    rng = random.Random(seed)
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=_sort_key)
        rng.shuffle(bucket)
        n_val = min(len(bucket) - 1, max(1, round(len(bucket) * val_ratio))) if len(bucket) > 1 else 0
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])
    return sorted(train, key=_sort_key), sorted(val, key=_sort_key)


def _count_records(records):
    return {
        "total": len(records),
        "labels": dict(sorted(Counter(str(r["post"]["source"]["label"]) for r in records).items())),
        "domains": dict(sorted(Counter(r["domain"] for r in records).items())),
        "datasets": dict(sorted(Counter(r["dataset"] for r in records).items())),
    }


def build_ood_manifests(dataset_root="dataset", protocol="cross_dataset", sources=None,
                        target=None, holdouts=None, seeds=(0, 1, 2, 3, 4), val_ratio=0.2,
                        pheme_archive="external_data/PHEME_veracity.tar.bz2", workers=8):
    """Build multiple seeds while reading source data and event metadata once."""
    if protocol not in PROTOCOLS:
        raise ValueError("Unsupported OOD protocol: {}".format(protocol))
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    seeds = list(seeds)
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Repeated seeds are not allowed")
    holdouts = sorted(set(str(h).strip() for h in (holdouts or [])))
    if protocol == "cross_dataset":
        if not sources or not target:
            raise ValueError("cross_dataset requires source dataset(s) and a target dataset")
        sources = [_canonical_dataset(s) for s in sources]
        target = _canonical_dataset(target)
        if len(set(sources)) != len(sources) or target in sources:
            raise ValueError("Sources must be unique and exclude the target dataset")
        if holdouts:
            raise ValueError("cross_dataset does not accept holdout themes/events")
    else:
        expected = "DRWeibo" if protocol == "drweibo_theme" else "Pheme"
        if sources and list(map(_canonical_dataset, sources)) != [expected]:
            raise ValueError("{} uses only {}".format(protocol, expected))
        if target and _canonical_dataset(target) != expected:
            raise ValueError("{} target must be {}".format(protocol, expected))
        sources, target = [expected], expected
        if not holdouts or "" in holdouts:
            raise ValueError("{} requires nonempty holdout domain(s)".format(protocol))
    sources = sorted(sources)
    roots, records = {}, {}
    for dataset in sorted(set(sources + [target])):
        directory, posts = _load_dataset(dataset, dataset_root, workers)
        roots[dataset] = str(directory)
        records[dataset] = posts
    excluded_missing = 0
    archive_sha = None
    if protocol == "cross_dataset":
        source = [post for d in sources for post in records[d]]
        test = records[target]
    else:
        event_map = None
        if protocol == "pheme_event":
            archive_path = Path(pheme_archive).expanduser().resolve()
            event_map = read_pheme_events(archive_path)
            # Membership in the event map, not archive bytes, determines splits.
            archive_sha = hashlib.sha256(_json_bytes(event_map)).hexdigest()
        known = []
        for record in records[target]:
            root = record["post"]["source"]
            domain = event_map.get(str(root["tweet id"])) if event_map is not None else root.get("theme")
            domain = str(domain).strip() if domain is not None else ""
            if domain.casefold() in _UNKNOWN_THEMES:
                excluded_missing += 1
                continue
            record["domain"] = domain
            known.append(record)
        available = {r["domain"] for r in known}
        missing = set(holdouts) - available
        if missing:
            raise ValueError("Unknown holdout domains {}; available: {}".format(sorted(missing), sorted(available)))
        source = [r for r in known if r["domain"] not in holdouts]
        test = [r for r in known if r["domain"] in holdouts]
    input_counts = {"source": len(source), "target": len(test), "excluded_missing_domain": excluded_missing}
    source, test, dedup = _deduplicate(source, test)
    source_domains = {domain: i for i, domain in enumerate(sorted({r["domain"] for r in source}))}
    if len({r["post"]["source"]["label"] for r in source}) < 2:
        raise ValueError("Source data must contain both labels after deduplication")
    manifests = []
    for seed in seeds:
        train, val = _source_split(source, seed, val_ratio)
        splits = {"train": train, "val": val, "test": test}
        if any(not value for value in splits.values()):
            raise ValueError("OOD train, validation, and test splits must all be nonempty after deduplication")
        manifest = {
            "version": VERSION,
            "protocol": protocol,
            "seed": seed,
            "source_datasets": sources,
            "target_dataset": target,
            "source_roots": roots,
            "source_domains": source_domains,
            "holdout_domains": holdouts,
            "val_ratio": val_ratio,
            "selection": "source_domain_label_stratified; target_labels_for_evaluation_only",
            "deduplication": "source_first_connected_components_platform_id_or_normalized_root_text; overlap_removed_from_test",
            "splits": {name: [{k: r[k] for k in ("dataset", "file", "sha256", "domain")} for r in values]
                       for name, values in splits.items()},
            "diagnostics": {
                "input": input_counts,
                "deduplication": dedup,
                "splits": {name: _count_records(values) for name, values in splits.items()},
                "single_class_test": len({r["post"]["source"]["label"] for r in test}) < 2,
            },
        }
        if archive_sha is not None:
            manifest["event_mapping_sha256"] = archive_sha
        manifest["fingerprint"] = manifest_digest(manifest)
        manifests.append(manifest)
    return manifests


def build_ood_manifest(dataset_root="dataset", protocol="cross_dataset", sources=None,
                       target=None, holdouts=None, seed=0, val_ratio=0.2,
                       pheme_archive="external_data/PHEME_veracity.tar.bz2", workers=8):
    """Single-seed equivalent of :func:`build_ood_manifests`."""
    return build_ood_manifests(dataset_root, protocol, sources, target, holdouts,
                               [seed], val_ratio, pheme_archive, workers)[0]


def write_ood_manifest(manifest, path, overwrite=False):
    """Write atomically; changing an existing manifest requires explicit opt-in."""
    path = Path(path).expanduser()
    if manifest.get("fingerprint") != manifest_digest(manifest):
        raise ValueError("Manifest fingerprint mismatch before writing")
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old == manifest:
            return path
        if not overwrite:
            raise FileExistsError("Different manifest already exists: {}; use --overwrite explicitly".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive staging filename avoids clobbering another writer's temporary file.
    import os
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        if path.exists() and not overwrite:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old != manifest:
                raise FileExistsError("Different manifest appeared while writing: {}".format(path))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def resolve_ood_manifest_path(path, seed=None):
    path = str(path)
    if "{seed}" in path:
        if seed is None:
            raise ValueError("A seed is required for an ood_manifest containing {seed}")
        path = path.replace("{seed}", str(seed))
    if "{" in path or "}" in path:
        raise ValueError("Only the {seed} placeholder is supported in ood_manifest")
    return Path(path).expanduser().resolve()


def _validate_manifest_structure(manifest, seed=None):
    if manifest.get("version") != VERSION or manifest.get("protocol") not in PROTOCOLS:
        raise ValueError("Unsupported OOD manifest version or protocol")
    if manifest.get("fingerprint") != manifest_digest(manifest):
        raise ValueError("OOD manifest fingerprint mismatch; regenerate the manifest")
    if seed is not None and manifest.get("seed") != seed:
        raise ValueError("OOD manifest seed {} differs from requested seed {}".format(manifest.get("seed"), seed))
    stored_seed = manifest.get("seed")
    if isinstance(stored_seed, bool) or not isinstance(stored_seed, int) or stored_seed < 0:
        raise ValueError("OOD manifest seed must be a nonnegative integer")
    if not isinstance(manifest.get("val_ratio"), (int, float)) or not 0 < manifest["val_ratio"] < 1:
        raise ValueError("OOD manifest val_ratio must be strictly between 0 and 1")
    if not isinstance(manifest.get("source_roots"), dict):
        raise ValueError("OOD manifest requires source_roots")
    sources = manifest.get("source_datasets")
    target = manifest.get("target_dataset")
    if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)) or target not in DATASETS or any(d not in DATASETS for d in sources):
        raise ValueError("OOD manifest source/target datasets are invalid")
    domains = manifest.get("source_domains")
    if (not isinstance(domains, dict) or not domains
            or any(not isinstance(k, str) or not k.strip() for k in domains)
            or any(type(v) is not int for v in domains.values())
            or sorted(domains.values()) != list(range(len(domains)))):
        raise ValueError("OOD source_domains must map domains to consecutive integer IDs")
    if manifest["protocol"] == "cross_dataset" and target in sources:
        raise ValueError("Cross-dataset target must be excluded from sources")
    if manifest["protocol"] != "cross_dataset":
        expected = "DRWeibo" if manifest["protocol"] == "drweibo_theme" else "Pheme"
        holdouts = manifest.get("holdout_domains")
        if sources != [expected] or target != expected or not isinstance(holdouts, list) or not holdouts:
            raise ValueError("Held-out OOD protocol has invalid datasets or domains")
        if any(not isinstance(h, str) or h.casefold() in _UNKNOWN_THEMES for h in holdouts):
            raise ValueError("Held-out OOD protocol contains missing/unknown domains")
    seen_files = set()
    for split in ("train", "val", "test"):
        records = manifest.get("splits", {}).get(split)
        if not isinstance(records, list) or not records:
            raise ValueError("OOD split {} is missing or empty".format(split))
        for record in records:
            if not isinstance(record, dict) or not all(k in record for k in ("dataset", "file", "sha256", "domain")):
                raise ValueError("Malformed OOD record in {}".format(split))
            if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
                raise ValueError("Invalid file SHA-256 in {}".format(split))
            if not isinstance(record["file"], str) or not isinstance(record["domain"], str) or not record["domain"].strip():
                raise ValueError("OOD record file/domain must be nonempty strings")
            _record_path(manifest, record)
            if manifest["protocol"] == "cross_dataset" and record["domain"] != record["dataset"]:
                raise ValueError("Cross-dataset domain must equal its dataset name")
            key = record["dataset"], record["file"]
            if key in seen_files:
                raise ValueError("Repeated file across OOD manifest: {}".format(key))
            seen_files.add(key)
            if split == "test" and record["dataset"] != target:
                raise ValueError("Target split includes a different dataset")
            if split != "test" and (record["dataset"] not in sources or record["domain"] not in domains):
                raise ValueError("Source split includes an unknown dataset/domain")
            if manifest["protocol"] != "cross_dataset":
                held_out = record["domain"] in manifest.get("holdout_domains", [])
                if held_out != (split == "test"):
                    raise ValueError("Held-out domain appears on the wrong side of the OOD split")


def load_ood_manifest(path, seed=None, validate_files=True):
    """Resolve ``{seed}``, check schema/fingerprint and optionally read all files.

    Validation checks the entire post hash and train/val/test isolation by root
    platform/ID and nonempty normalized root text, including within each split.
    """
    resolved = resolve_ood_manifest_path(path, seed)
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    _validate_manifest_structure(manifest, seed)
    if validate_files:
        entries = [(split, record) for split in ("train", "val", "test") for record in manifest["splits"][split]]
        with ThreadPoolExecutor(max_workers=8) as pool:
            posts = pool.map(_read_post, [_record_path(manifest, r) for _, r in entries])
            seen = {}
            for (split, record), (post, digest) in zip(entries, posts):
                if digest != record["sha256"]:
                    raise ValueError("Source file changed: {}/{}; regenerate OOD manifests/cache".format(record["dataset"], record["file"]))
                if manifest["protocol"] == "drweibo_theme" and str(post["source"].get("theme", "")).strip() != record["domain"]:
                    raise ValueError("Theme metadata differs from source post")
                for key in _duplicate_keys(record["dataset"], post):
                    if key in seen:
                        raise ValueError("Duplicate root ID/text in OOD splits: {} and {}".format(seen[key], (split, record["dataset"], record["file"])))
                    seen[key] = (split, record["dataset"], record["file"])
    return manifest


def materialize_ood_posts(manifest, split):
    """Return namespaced file IDs and intact graph posts with source domain IDs.

    The returned tuple ID is safe as an output filename. The original value of
    ``post['source']['tweet id']`` and every graph/timestamp field are preserved.
    """
    if split not in ("train", "val", "test"):
        raise ValueError("Invalid OOD split: {}".format(split))
    records = manifest["splits"][split]
    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(_read_post, [_record_path(manifest, r) for r in records]))
    output = []
    for record, (post, digest) in zip(records, loaded):
        if digest != record["sha256"]:
            raise ValueError("Source file changed while materializing: {}".format(record["file"]))
        post = copy.deepcopy(post)
        post["source"]["domain_id"] = manifest["source_domains"].get(record["domain"], -1) if split != "test" else -1
        root_id = str(post["source"]["tweet id"])
        # Real data IDs are alphanumeric; encode unsafe IDs deterministically.
        safe_id = root_id if re.fullmatch(r"[A-Za-z0-9_-]+", root_id) else hashlib.sha256(root_id.encode()).hexdigest()
        output.append((record["dataset"] + "__" + safe_id, post))
    return output
