#!/usr/bin/env sh
# Drive the mAP-degradation motivation experiment: run the pretrained hybrid
# detector on clean + every (corruption, severity) and append mAP rows to
# results/neftci_map_degradation.csv. Mirrors test_gen1.sh.
#
# Each (corruption, severity) is a fresh process so there is no state leakage
# between runs. After the sweep it AUTO-RUNS summarize + check, so the whole
# motivation experiment is one command.
#
# Usage (flags override env; one-liner):
#   sh run_map_degradation.sh --test <gen1_dir> --output <out_dir> [--smoke] [--no-analysis]
#
# Env knobs (same spirit as test_gen1.sh):
#   GEN1_DATASET_DIR  path to Gen1 (default HybridDetection/gen1)
#   CHECKPOINT        ckpt path     (default HybridDetection/gen1_mAP36.ckpt)
#   GPU_ID            GPU index     (default 0)
#   USE_TEST_SET      true|false    (default true -- the held-out test split)
#   SMOKE             1 => only a few batches per run (pipeline check, cheap)
#   CORRUPTIONS       space list    (default: all 6)
#   SEVERITIES        space list    (default: 1 2 3 4 5)
set -eu

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
UV_BIN="${UV_BIN:-uv}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

# ---- CLI flags (override env): --test <dir> --output <dir> --smoke --no-analysis ----
RUN_ANALYSIS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --test|--test-dir|--gen1|--data) GEN1_DATASET_DIR="$2"; shift 2 ;;
    --output|--output-dir|--out)     OUTPUT_DIR="$2"; shift 2 ;;
    --ckpt|--checkpoint)             CHECKPOINT="$2"; shift 2 ;;
    --gpu)                           GPU_ID="$2"; shift 2 ;;
    --smoke)                         SMOKE=1; shift ;;
    --no-analysis)                   RUN_ANALYSIS=0; shift ;;
    -h|--help)
      echo "Usage: sh run_map_degradation.sh --test <gen1_dir> --output <out_dir> [--smoke] [--no-analysis]"
      exit 0 ;;
    *) echo "[run_map_degradation] ignoring unknown arg: $1" >&2; shift ;;
  esac
done

# Dataset: explicit GEN1_DATASET_DIR wins; else HybridDetection/gen1; else the
# project-root gen1 (the usual layout, since HybridDetection/ is a sub-dir).
if [ -n "${GEN1_DATASET_DIR:-}" ]; then
  DATA_DIR="${GEN1_DATASET_DIR}"
elif [ -d "${ROOT_DIR}/gen1" ]; then
  DATA_DIR="${ROOT_DIR}/gen1"
else
  DATA_DIR="${ROOT_DIR}/../gen1"
fi
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/gen1_mAP36.ckpt}"
EXPERIMENT_CFG="${EXPERIMENT_CFG:-no_lstm}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
USE_TEST_SET="${USE_TEST_SET:-true}"
CHECKPOINT_STRICT="${CHECKPOINT_STRICT:-false}"
CORRUPTIONS="${CORRUPTIONS:-hot_pixel event_flood temporal_jitter event_rate_shift polarity_flip spatial_dropout}"
SEVERITIES="${SEVERITIES:-1 2 3 4 5}"
# Everything (CSV, summary, table, figures) is saved under outputs/. Default:
# project vmem_benchmark/outputs/results; --output overrides.
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESULTS_OUT="${OUTPUT_DIR:-${ROOT_DIR}/../vmem_benchmark/outputs/results}"

CHECKPOINT_HYDRA="$(printf '%s' "$CHECKPOINT" | sed 's/=/\\=/g')"

run_one() {
  corruption="$1"; severity="$2"
  set -- \
    "dataset=gen1" \
    "+experiment/gen1=${EXPERIMENT_CFG}.yaml" \
    "dataset.path=${DATA_DIR}" \
    "checkpoint=${CHECKPOINT_HYDRA}" \
    "checkpoint_load_strict=${CHECKPOINT_STRICT}" \
    "use_test_set=${USE_TEST_SET}" \
    "hardware.gpus=${GPU_ID}" \
    "batch_size.eval=${BATCH_SIZE_EVAL}" \
    "hardware.num_workers.eval=${EVAL_WORKERS}" \
    "+corruption=${corruption}" \
    "+severity=${severity}"
  set -- "$@" "+output_dir=${RESULTS_OUT}"
  if [ "${SMOKE:-0}" = "1" ]; then
    if [ "${USE_TEST_SET}" = "true" ]; then
      set -- "$@" "+limit_test_batches=4"
    else
      set -- "$@" "+limit_val_batches=4"
    fi
  fi
  echo ">>> ${corruption} sev${severity}"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "$UV_BIN run python validation_corrupt.py $*"; return 0
  fi
  ( cd "$ROOT_DIR" && "$UV_BIN" run python validation_corrupt.py "$@" )
}

# clean baseline first (severity is ignored for the pass-through, fixed to 0)
run_one clean 0 || true
for c in $CORRUPTIONS; do
  for s in $SEVERITIES; do
    run_one "$c" "$s" || echo "!!! ${c} sev${s} failed, continuing"
  done
done

# ---- auto-summarize + check (the standalone analysis scripts: pandas/matplotlib) ----
CSV="${RESULTS_OUT}/neftci_map_degradation.csv"
if [ "${RUN_ANALYSIS}" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  if [ -z "${ANALYSIS_PY:-}" ]; then
    if   [ -x "${ROOT_DIR}/../vmem_benchmark/.venv/Scripts/python.exe" ]; then ANALYSIS_PY="${ROOT_DIR}/../vmem_benchmark/.venv/Scripts/python.exe"
    elif [ -x "${ROOT_DIR}/../vmem_benchmark/.venv/bin/python" ]; then         ANALYSIS_PY="${ROOT_DIR}/../vmem_benchmark/.venv/bin/python"
    else                                                                       ANALYSIS_PY="python"
    fi
  fi
  echo "Summarizing + checking -> ${RESULTS_OUT}"
  "$ANALYSIS_PY" "${ROOT_DIR}/../analysis/summarize_map_degradation.py" --input "$CSV" --output-dir "$RESULTS_OUT" || echo "[!] summarize step failed"
  if [ -n "${OUTPUT_DIR}" ]; then
    "$ANALYSIS_PY" "${ROOT_DIR}/../analysis/check_map_degradation.py" --map-csv "$CSV" --output-dir "$RESULTS_OUT" || echo "[!] check step failed"
  else
    "$ANALYSIS_PY" "${ROOT_DIR}/../analysis/check_map_degradation.py" --map-csv "$CSV" || echo "[!] check step failed"
  fi
fi
echo "Done -> ${RESULTS_OUT}"
