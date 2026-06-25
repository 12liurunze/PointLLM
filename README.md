# Semantic Octree Compression Ablation

This project is an isolated PointLLM EAGLE compression project.

Included:

- Semantic Octree point-token selection.
- Geometry, semantic, and summary components.
- Training-data generation.
- Standard single-step EAGLE head training.
- FP32 autoregressive versus EAGLE evaluation.
- Point-cloud compression visualization.
- Full compression ablation runner.

Excluded:

- Rollout loss and adaptive rollout.
- Coverage-aware rollout.
- Token-level rollout.
- Point-grounded corrective rollout.
- Global-token experiments.
- Boundary and MAT-boost training modifications.

## Project path

Place this directory anywhere. The examples below assume:

```text
/lnt/workspace/jiaming.fjm/lrz/eagle-eye-semantic-octree-ablation-a100
```


## Environment and paths

The scripts now default to the requested 4 x A100 layout:

```text
POINTLLM_REPO=/lnt/workspace/jiaming.fjm/lrz/pointLLM
BASE_MODEL=/lnt/workspace/jiaming.fjm/lrz/point7B_v1.1
POINT_CLOUD_DATA=/lnt/workspace/jiaming.fjm/lrz/pointLLM/data/objaverse_data/8192_npy
ANNO_DIR=/lnt/workspace/jiaming.fjm/lrz/pointLLM/data/anno_data
CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
```

Install the CUDA 12.8-oriented environment with:

```bash
cd EAGLE_EYE
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Training is launched through `python -m torch.distributed.run --nproc_per_node=${NUM_GPUS}`.
`BATCH_SIZE` is interpreted as per-GPU batch size; the default is 8, so the default global batch size is 32 on 4 GPUs.
Data generation is also split into four independent ranges, one per GPU.
FP32 speed evaluation intentionally remains single-GPU so timing stays comparable
to the autoregressive baseline.

## Full ablation

```bash
cd /lnt/workspace/jiaming.fjm/lrz/eagle-eye-semantic-octree-ablation-a100

nohup bash scripts/run_semantic_octree_full_ablation.sh \
  > /lnt/workspace/jiaming.fjm/lrz/semantic_octree_full_ablation.log 2>&1 &
```

The default run performs 18 independently trained experiments:

- Uncompressed baseline.
- Seven non-empty combinations of geometry, semantic, and summary.
- Keep ratios: 1%, 3%, 5%, 10%, 25%.
- Summary counts: 0, 4, 8, 16.
- Octree depths: 2, 3, 4, 5.

The standard `5% + 8 summaries + depth 4` configuration is reused as the
center point and is not duplicated in each sweep.

Run selected groups:

```bash
ABLATION_GROUPS=baseline,components \
bash scripts/run_semantic_octree_full_ablation.sh
```

Useful overrides:

```bash
END=10000
NUM_EPOCHS=8
BATCH_SIZE=8   # per GPU; 4 GPUs => global batch 32
KEEP_DATA=0
KEEP_HEADS=1
```

Results are written to:

```text
results/semantic_octree_ablation/<run_id>/
```

Training data and heads are written under:

```text
/lnt/workspace/jiaming.fjm/lrz/semantic_octree_ablation_runs/<run_id>/
```

## Smoke test

```bash
START=0 END=8 EVAL_START=0 EVAL_END=2 \
NUM_EPOCHS=1 BATCH_SIZE=2 \
ABLATION_GROUPS=components \
bash scripts/run_semantic_octree_full_ablation.sh
```
