#!/usr/bin/env bash
# Paper plan (h256x2, 269k bits, seeds 42-46, diagnostics on everywhere):
#   Stage 1  population ladder  P in {256,512,1024,2048}, rank, 2000 it
#   Stage 2  shaping comparison standardize vs rank: std twin cells at
#            P=2048, 2000 it (rank side = stage 1's P=2048 cells)
#   Stage 3  long runs          P=2048, rank, 5000 it
#   Stage 4  STE comparison     h256x2, he scale, no-input-quant, 100 epochs:
#            adam lr 1e-3 + sgd lr {0.003,0.01,0.03}, 5 seeds each
# lr rule: lr = 100/P. Each run skips itself if its output JSON exists.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/data/ste/venv/bin/python}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "interpreter not found: $PY" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
COMMON_ES="--hidden 256 --layers 2 --scale he --activation relu --fitness loss --loss ce --sigma 0 --table-rows 0 --chunk 512 --grad-chunk 64 --pop-chunk 128 --eval-every 10 --diagnostics"

lr_for() {
  case $1 in
    256)  echo 0.390625 ;;
    512)  echo 0.1953125 ;;
    1024) echo 0.09765625 ;;
    2048) echo 0.048828125 ;;
  esac
}

echo "=== stage 1: population ladder (rank, 2000 it) $(date) ==="
for P in 256 512 1024 2048; do
  LR=$(lr_for "$P")
  for S in 42 43 44 45 46; do
    OUT="results/es_h256x2_he_relu_p${P}_lr${LR}_shrank_seed${S}_t0_diag2000.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    echo "--- $(date +%H:%M:%S) ladder P=$P seed=$S"
    $PY experiments/run_es.py $COMMON_ES --shaping rank \
      --population "$P" --lr "$LR" --iterations 2000 --seed "$S" --out "$OUT" \
      || echo "FAILED ladder P=$P seed=$S"
  done
done

echo "=== stage 2: shaping comparison (standardize, P=2048, 2000 it) $(date) ==="
for S in 42 43 44 45 46; do
  OUT="results/es_h256x2_he_relu_p2048_lr0.048828125_shstandardize_seed${S}_t0_diag2000.json"
  if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
  echo "--- $(date +%H:%M:%S) shaping std P=2048 seed=$S"
  $PY experiments/run_es.py $COMMON_ES --shaping standardize \
    --population 2048 --lr 0.048828125 --iterations 2000 --seed "$S" --out "$OUT" \
    || echo "FAILED shaping std seed=$S"
done

echo "=== stage 3: long runs (rank, P=2048, 5000 it) $(date) ==="
for S in 42 43 44 45 46; do
  OUT="results/es_h256x2_he_relu_p2048_lr0.048828125_shrank_seed${S}_t0_diag5000.json"
  if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
  echo "--- $(date +%H:%M:%S) long P=2048 seed=$S"
  $PY experiments/run_es.py $COMMON_ES --shaping rank \
    --population 2048 --lr 0.048828125 --iterations 5000 --seed "$S" --out "$OUT" \
    || echo "FAILED long seed=$S"
done

echo "=== stage 4: STE comparison (h256x2, he, noiq, 100 epochs) $(date) ==="
for S in 42 43 44 45 46; do
  OUT="results/ste_h256x2_he_noiq_adam_seed${S}.json"
  if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
  echo "--- $(date +%H:%M:%S) STE adam seed=$S"
  TF_USE_LEGACY_KERAS=1 $PY experiments/run_ste.py \
    --hidden 256 --layers 2 --scale he --no-input-quant \
    --optimizer adam --learning-rate 0.001 --epochs 100 --eval-every 50 \
    --seed "$S" --out "$OUT" || echo "FAILED STE adam seed=$S"
done
for LR in 0.003 0.01 0.03; do
  for S in 42 43 44 45 46; do
    OUT="results/ste_h256x2_he_noiq_sgd_lr${LR}_seed${S}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    echo "--- $(date +%H:%M:%S) STE sgd lr=$LR seed=$S"
    TF_USE_LEGACY_KERAS=1 $PY experiments/run_ste.py \
      --hidden 256 --layers 2 --scale he --no-input-quant \
      --optimizer sgd --learning-rate "$LR" --epochs 100 --eval-every 50 \
      --seed "$S" --out "$OUT" || echo "FAILED STE sgd lr=$LR seed=$S"
  done
done

echo "=== plan complete $(date) ==="
