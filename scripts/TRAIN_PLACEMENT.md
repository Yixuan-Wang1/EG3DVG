# Canonical placement dataset training

The custom loader consumes the five manifests in `dataset_info/`, the language
splits in `splits/`, and scene files below `/data/jiajun.xie/3D_Box/data`.
Geometry stored in centimetres is converted to metres automatically.

Expected model assets:

```text
/data/jiajun.xie/3D_Box/data/eg3dvg_assets/
├── pointnet_backbone.pth
└── roberta-base/
    ├── config.json
    ├── pytorch_model.bin
    ├── merges.txt
    └── vocab.json
```

Run a short single-GPU smoke test first:

```bash
cd /path/to/EG3DVG
GPUS=0 NPROC_PER_NODE=1 BATCH_SIZE=1 NUM_WORKERS=0 MAX_EPOCH=1 \
  bash scripts/train_placement.sh --debug --print_freq 1
```

`--debug` limits each custom split to 128 records. Once data loading, forward,
and loss computation succeed, start the normal two-GPU run:

```bash
cd /path/to/EG3DVG
bash scripts/train_placement.sh
```

Common overrides:

```bash
GPUS=0,1,2,3 NPROC_PER_NODE=4 BATCH_SIZE=2 NUM_WORKERS=16 \
MODEL_ROOT=/path/to/eg3dvg_assets OUTPUT_ROOT=/path/to/output \
  bash scripts/train_placement.sh --checkpoint_path /path/to/checkpoint.pth
```

`BATCH_SIZE` is per GPU. The split JSON files group by source and sample, so
frames cannot leak between train, validation, and test through repeated language
instructions.
