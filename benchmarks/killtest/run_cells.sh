#!/bin/bash
# Fresh-process-per-cell kill-test driver (protocol amendment 2026-07-15).
set -e
cd "$(dirname "$0")/../.."
SMCJAX_PROJECT="${SMCJAX_PROJECT:-../smcjax}"
for w in lgssm sv track track_full; do
  for n in 10000 100000 1000000; do
    echo "=== jax $w $n ==="
    JAX_PLATFORMS=cpu uv run --project "${SMCJAX_PROJECT}" \
      python benchmarks/killtest/jax_side.py "$w" "$n"
  done
done
for w in lgssm sv track track_full; do
  for n in 10000 100000 1000000; do
    echo "=== mlx $w $n ==="
    uv run python benchmarks/killtest/mlx_side.py "$w" "$n"
  done
done
python3 - <<'PY'
import json, pathlib
data = pathlib.Path('benchmarks/data')
for side in ('jax', 'mlx'):
    cells = {}
    for f in sorted((data / 'cells').glob(f'{side}_*.json')):
        _, w, n = f.stem.split('_', 1)[0], '_'.join(f.stem.split('_')[1:-1]), f.stem.split('_')[-1]
        cells[f'{w}/{n}'] = json.loads(f.read_text())
    base = json.loads((data / f'{side}_results.json').read_text())
    base['cells'] = cells
    (data / f'{side}_results.json').write_text(json.dumps(base))
print('merged')
PY
