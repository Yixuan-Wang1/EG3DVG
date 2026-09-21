"""Validate placement split files against the local dataset manifests."""

import argparse
import json
import re
from pathlib import Path


def read(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, default=Path("splits"))
    parser.add_argument("--manifest-root", type=Path, default=Path("dataset_info"))
    args = parser.parse_args()
    summary = read(args.split_root / "manifest.json")
    aliases = {"omni": "omni_filter"}
    sample_counts = {}
    for path in args.manifest_root.glob("*/manifest.json"):
        for sample in read(path)["samples"]:
            sample_counts[(path.parent.name, sample["sample_id"])] = sample["object_count"]

    item_sets, group_sets, all_items = {}, {}, []
    failed = False
    for split in ("train", "valid", "test"):
        payload = read(args.split_root / f"{split}.json")
        items = payload["items"]
        all_items.extend(items)
        missing, bad_objects = [], []
        for item in items:
            source = aliases.get(item["source_name"], item["source_name"])
            key = (source, item["sample_id"])
            if key not in sample_counts:
                missing.append(item["item_id"])
                continue
            # Object IDs retain source indices and need not be contiguous, so an
            # obj_5 can be valid in a four-object filtered scene. Exact membership
            # can only be checked against the server-side sample JSON.
            match = re.fullmatch(r"obj_\d+(?:_[A-Za-z0-9_]+)?", item["object_id"])
            if match is None:
                bad_objects.append(item["item_id"])
        item_sets[split] = {item["item_id"] for item in items}
        group_sets[split] = {(item["source_name"], item["sample_id"]) for item in items}
        expected_items = summary["split_item_counts"][split]
        expected_groups = summary["split_group_counts"][split]
        duplicate_count = len(items) - len(item_sets[split])
        print(f"{split}: actual={len(items)}, declared={payload['item_count']}, "
              f"manifest={expected_items}, groups={len(group_sets[split])}/{expected_groups}, "
              f"missing_samples={len(missing)}, bad_object_ids={len(bad_objects)}, "
              f"duplicate_item_ids={duplicate_count}")
        failed |= bool(missing or bad_objects or duplicate_count)
        failed |= len(items) != payload["item_count"] or len(items) != expected_items
        failed |= len(group_sets[split]) != expected_groups

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        item_overlap = item_sets[left] & item_sets[right]
        sample_overlap = group_sets[left] & group_sets[right]
        print(f"{left}/{right}: item_overlap={len(item_overlap)}, sample_overlap={len(sample_overlap)}")
        failed |= bool(item_overlap or sample_overlap)

    for source in ("dopose", "hope", "housecat", "omni", "ycbv"):
        labels = {item["label_index"] for item in all_items if item["source_name"] == source}
        gaps = [value for value in range(min(labels), max(labels) + 1) if value not in labels]
        self_refs = sum(item["source_name"] == source and "itself" in item["instruction"].lower()
                        for item in all_items)
        print(f"{source}: labels={len(labels)}, range={min(labels)}..{max(labels)}, "
              f"label_gaps={len(gaps)}, first_gaps={gaps[:10]}, self_refs={self_refs}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
