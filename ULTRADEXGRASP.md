# UltraDexGrasp In DP3 Workspace

This workspace is separated from `UltraDexGrasp` and uses `uv` to manage the DP3 training environment.

## 1. Sync the environment

```bash
cd /home/ym/fzb/dp3-workspace
uv sync
```

## 2. Import UltraDexGrasp demonstrations

This converts `episode_*.npz` files into DP3-ready zarr datasets under `data/ultradexgrasp/`.

```bash
uv run python scripts/import_ultradexgrasp_data.py \
  --source-npz-root /home/ym/fzb/UltraDexGrasp/outputs/batch_run \
  --output-root /home/ym/fzb/dp3-workspace/data/ultradexgrasp \
  --num-points 1024
```

Generated datasets:

- `data/ultradexgrasp/single_left.zarr`
- `data/ultradexgrasp/single_right.zarr`
- `data/ultradexgrasp/bimanual.zarr`

## 3. Train

```bash
uv run bash scripts/train_ultradexgrasp.sh single_left dp3 0 0
uv run bash scripts/train_ultradexgrasp.sh single_right dp3 0 0
uv run bash scripts/train_ultradexgrasp.sh bimanual dp3 0 0
```

Arguments:

1. group: `single_left` | `single_right` | `bimanual`
2. algorithm: `dp3` | `simple_dp3`
3. seed
4. gpu id
