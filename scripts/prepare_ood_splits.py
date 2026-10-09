#!/usr/bin/env python3
"""Create audited zero-shot OOD manifests without importing PyTorch."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.ood_splits import DATASETS, PROTOCOLS, build_ood_manifests, manifest_digest, write_ood_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="cross_dataset")
    parser.add_argument("--all-pairs", action="store_true", help="Generate all six directed single-source dataset pairs")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--target")
    parser.add_argument("--holdout", nargs="+", default=[], help="Exact theme or event name(s) to hold out")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--pheme-archive", default="external_data/PHEME_veracity.tar.bz2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.all_pairs and (args.protocol != "cross_dataset" or args.sources or args.target or args.holdout):
        parser.error("--all-pairs is for cross_dataset and cannot be combined with --sources, --target, or --holdout")
    pairs = [(source, target) for source in DATASETS for target in DATASETS if source != target] if args.all_pairs else [(None, None)]
    audit = {"version": 1, "protocol": args.protocol, "manifests": []}
    for source, target in pairs:
        manifests = build_ood_manifests(
            dataset_root=args.dataset_root, protocol=args.protocol,
            sources=[source] if args.all_pairs else args.sources,
            target=target if args.all_pairs else args.target,
            holdouts=args.holdout, seeds=args.seeds,
            val_ratio=args.val_ratio, pheme_archive=args.pheme_archive, workers=args.workers,
        )
        output = Path(args.output_dir) / (source.lower() + "_to_" + target.lower()) if args.all_pairs else Path(args.output_dir)
        for manifest in manifests:
            path = output / "seed_{}.json".format(manifest["seed"])
            write_ood_manifest(manifest, path, overwrite=args.overwrite)
            entry = {"manifest": str(path), "source_datasets": manifest["source_datasets"],
                     "target_dataset": manifest["target_dataset"], "seed": manifest["seed"],
                     "fingerprint": manifest["fingerprint"], "diagnostics": manifest["diagnostics"]}
            audit["manifests"].append(entry)
            print(json.dumps(entry, ensure_ascii=False))
    audit["fingerprint"] = manifest_digest(audit)
    write_ood_manifest(audit, Path(args.output_dir) / "audit_summary.json", overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
