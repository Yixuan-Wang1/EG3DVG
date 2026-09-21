"""Dataset adapter for canonical_placement_scene/v1 data."""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

MAX_NUM_OBJ = 132
MAX_TOKENS = 256


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def source_directory(root, source):
    path = Path(root) / ({"omni": "omni_filter"}.get(source, source))
    if not path.is_dir():
        raise FileNotFoundError(f"Missing source directory for {source!r}: {path}")
    return path


def world_aabb(obj, scale):
    bounds = np.asarray(obj["bbox3d_canonical"], dtype=np.float32)
    corners = np.stack(np.meshgrid(*zip(bounds[:3], bounds[3:]), indexing="ij"), -1).reshape(-1, 3)
    pose = np.asarray(obj["pose_world"], dtype=np.float32)
    corners = (corners @ pose[:3, :3].T + pose[:3, 3]) * scale
    low, high = corners.min(0), corners.max(0)
    return np.concatenate(((low + high) / 2, high - low)).astype(np.float32)


def points_in_object(xyz_m, obj, scale):
    """Test points against the original oriented box, not its loose world AABB."""
    pose = np.asarray(obj["pose_world"], dtype=np.float32)
    world = xyz_m / scale
    local = (world - pose[:3, 3]) @ pose[:3, :3]
    bounds = np.asarray(obj["bbox3d_canonical"], dtype=np.float32)
    return np.all((local >= bounds[:3] - 1e-4) & (local <= bounds[3:] + 1e-4), axis=1)


def read_ply(path, scale):
    vertices = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")]).astype(np.float32) * scale
    names = set(vertices.dtype.names)
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack([vertices[key] for key in ("red", "green", "blue")]).astype(np.float32)
        if rgb.max(initial=0) > 1.5:
            rgb /= 255.0
    else:
        rgb = np.zeros_like(xyz)
    return xyz, rgb


def positive_map(tokenizer, instruction, class_name):
    result = np.zeros((MAX_NUM_OBJ, MAX_TOKENS), dtype=np.float32)
    caption, phrase = instruction.lower(), class_name.replace("_", " ").lower()
    start = caption.find(phrase)
    if start >= 0:
        end = start + len(phrase)
    else:
        spans = [(caption.find(word), len(word)) for word in re.findall(r"[a-z0-9]+", phrase)]
        spans = [(pos, length) for pos, length in spans if pos >= 0]
        start = min((pos for pos, _ in spans), default=0)
        end = max((pos + length for pos, length in spans), default=1)
    encoded = tokenizer(instruction, truncation=True, max_length=MAX_TOKENS,
                        return_offsets_mapping=True)
    for token, (left, right) in enumerate(encoded["offset_mapping"][:MAX_TOKENS]):
        if right > start and left < end:
            result[0, token] = 1
    if result[0].sum():
        result[0] /= result[0].sum()
    return result


class Placement3DDataset(Dataset):
    def __init__(self, data_root, split_root, manifest_root, split, tokenizer_root, num_points=50000,
                 use_color=True, superpoint_voxel_size=0.03, seed=0, limit=None):
        self.data_root, self.split_root = Path(data_root), Path(split_root)
        self.manifest_root = Path(manifest_root)
        self.split = "valid" if split == "val" else split
        self.items = load_json(self.split_root / f"{self.split}.json")["items"]
        if limit is not None:
            self.items = self.items[:limit]
        self.num_points, self.use_color = num_points, use_color
        self.superpoint_voxel_size, self.seed = superpoint_voxel_size, seed
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            os.path.join(tokenizer_root, "roberta-base"), local_files_only=True)
        self.manifests = {}

    def __len__(self):
        return len(self.items)

    def sample_path(self, source, sample_id):
        if source not in self.manifests:
            directory = source_directory(self.data_root, source)
            manifest_path = self.manifest_root / ({"omni": "omni_filter"}.get(source, source)) / "manifest.json"
            manifest = load_json(manifest_path)
            self.manifests[source] = (directory, {x["sample_id"]: x["sample_path"] for x in manifest["samples"]})
        directory, entries = self.manifests[source]
        if sample_id not in entries:
            raise KeyError(f"{sample_id!r} absent from manifest for {source!r}")
        return directory, directory / entries[sample_id]

    def __getitem__(self, index):
        item = self.items[index]
        directory, sample_path = self.sample_path(item["source_name"], item["sample_id"])
        scene = load_json(sample_path)
        if scene.get("schema_version") != "canonical_placement_scene/v1":
            raise ValueError(f"Unsupported schema in {sample_path}")
        scale = 0.01 if scene.get("unit", "cm") == "cm" else 1.0
        cloud_path = scene.get("voxel_point_cloud_path") or scene["point_cloud_path"]
        xyz, rgb = read_ply(directory / cloud_path, scale)
        if not len(xyz):
            raise ValueError(f"Empty point cloud: {directory / cloud_path}")
        rng = np.random.RandomState(self.seed + index) if self.split != "train" else np.random
        selected = rng.choice(len(xyz), self.num_points, replace=len(xyz) < self.num_points)
        xyz, rgb = xyz[selected], rgb[selected]
        mean_rgb = np.array([109.8, 97.2, 83.8], dtype=np.float32) / 256
        cloud = np.concatenate((xyz, rgb - mean_rgb), 1) if self.use_color else xyz

        objects = scene["objects"][:MAX_NUM_OBJ]
        boxes = np.stack([world_aabb(obj, scale) for obj in objects])
        lookup = {obj["obj_id"]: pos for pos, obj in enumerate(objects)}
        if item["object_id"] not in lookup:
            raise KeyError(f"Target {item['object_id']!r} absent from {sample_path}")
        target = lookup[item["object_id"]]

        all_boxes = np.zeros((MAX_NUM_OBJ, 6), np.float32)
        all_boxes[:len(boxes)] = boxes
        all_mask = np.zeros(MAX_NUM_OBJ, bool)
        all_mask[:len(boxes)] = True
        instances = np.full(self.num_points, -1, np.int64)
        for object_index, obj in enumerate(objects):
            inside = points_in_object(xyz, obj, scale)
            instances[(instances < 0) & inside] = object_index
        masks = np.zeros((MAX_NUM_OBJ, self.num_points), np.int64)
        masks[0] = instances == target
        _, superpoint = np.unique(np.floor(xyz / self.superpoint_voxel_size).astype(np.int64),
                                  axis=0, return_inverse=True)

        pos_map = positive_map(self.tokenizer, item["instruction"], objects[target]["class_name"])
        zero_map = np.zeros((MAX_NUM_OBJ, MAX_TOKENS), np.float32)
        box_labels = np.zeros(MAX_NUM_OBJ, np.float32)
        box_labels[0] = 1
        centers, sizes = np.zeros((MAX_NUM_OBJ, 3), np.float32), np.zeros((MAX_NUM_OBJ, 3), np.float32)
        centers[0], sizes[0] = boxes[target, :3], boxes[target, 3:]
        distractors = [i for i, obj in enumerate(objects)
                       if i != target and obj["class_name"] == objects[target]["class_name"]]
        empty_boxes, empty_mask = np.zeros((MAX_NUM_OBJ, 6), np.float32), np.zeros(MAX_NUM_OBJ, bool)

        return {
            "box_label_mask": box_labels, "center_label": centers,
            "sem_cls_label": np.zeros(MAX_NUM_OBJ, np.int64), "size_gts": sizes,
            "gt_masks": masks, "scan_ids": scene["sample_id"],
            "point_clouds": cloud.astype(np.float32), "og_color": rgb.astype(np.float32),
            "utterances": item["instruction"] + " . not mentioned",
            "language_dataset": item["source_name"],
            "tokens_positive": np.zeros((MAX_NUM_OBJ, 2), np.int64),
            "positive_map": pos_map, "modify_positive_map": zero_map.copy(),
            "pron_positive_map": zero_map.copy(), "other_entity_map": zero_map.copy(),
            "rel_positive_map": zero_map.copy(), "auxi_entity_positive_map": zero_map.copy(),
            "auxi_box": np.zeros((1, 6), np.float32), "relation": "none",
            "target_name": objects[target]["class_name"], "target_id": target,
            "point_instance_label": instances, "all_bboxes": all_boxes,
            "all_bbox_label_mask": all_mask, "all_class_ids": np.zeros(MAX_NUM_OBJ, np.int64),
            "distractor_ids": np.asarray((distractors + [-1] * 32)[:32], np.int64),
            "anchor_ids": np.full(32, -1, np.int64),
            "all_detected_boxes": empty_boxes,
            "all_detected_bbox_label_mask": empty_mask,
            "all_detected_class_ids": np.zeros(MAX_NUM_OBJ, np.int64),
            "all_detected_logits": np.zeros((MAX_NUM_OBJ, 485), np.float32),
            "is_view_dep": any(x in item["instruction"].lower() for x in ("front", "behind", "back", "left", "right")),
            "is_hard": len(distractors) > 1, "is_unique": not distractors, "target_cid": 0,
            "superpoint": torch.from_numpy(superpoint.astype(np.int64)),
            "negative_positive_map": zero_map.copy(),
            "negative_modify_positive_map": zero_map.copy(),
            "negative_pron_positive_map": zero_map.copy(),
            "negative_other_entity_map": zero_map.copy(),
            "negative_auxi_entity_positive_map": zero_map.copy(),
            "negative_rel_positive_map": zero_map.copy(), "negative_utterances": "none",
        }
