#!/usr/bin/env bash

# Run early-detection evaluation for all DRWeibo models and time points.
#
# Optional environment variables:
#   PROJECT_ROOT  Project directory (default: /public/wc/EIN-main)
#   PYTHON_BIN    Python executable (default: python)
#   DEVICE        Evaluation device (default: cuda:0)
#   LOG_ROOT      Output directory for per-run logs

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/wc/EIN-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEED=0
TIME_POINTS=(0h 1h 3h 5h 9h 12h 24h 36h)
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/drweibo_early_eval/$(date +%Y%m%d_%H%M%S)}"

MODEL_NAMES=(
  "our_model"
  "see"
  "ragcl"
  "bigcn"
  "EIN"
)

declare -A MODEL_SLUGS=(
  ["our_model"]="our_model"
  ["see"]="see"
  ["ragcl"]="ragcl"
  ["bigcn"]="bigcn"
  ["EIN"]="ein"
)

declare -A CONFIG_PATHS=(
  ["our_model"]="configs/EIN/DRWeibo_ResGCN_UncertaintySemanticChange_word2vec.yaml"
  ["see"]="configs/EIN/DRWeibo_SEEGraphMAE_word2vec.yaml"
  ["ragcl"]="configs/EIN/DRWeibo_RAGCL_ResGCN_word2vec.yaml"
  ["bigcn"]="configs/EIN/DRWeibo_RAGCL_BiGCN_word2vec.yaml"
  ["EIN"]="configs/EIN/DRWeibo.yaml"
)

declare -A CHECKPOINT_PATHS=(
  ["our_model"]="experiments/EIN/DRWeibo/test_duibixuexi_aux2.0_mlp_nouncertain_resgcn_nobias/seed_0/best_model.pth.m"
  ["see"]="experiments/EIN/DRWeibo/see_graphmae_directed_valloss_word2vec_manifest/seed_0/best_model.pth.m"
  ["ragcl"]="experiments/EIN/DRWeibo/ragcl_resgcn_directed_valloss_layer3_manifest/seed_0/best_model.pth.m"
  ["bigcn"]="experiments/EIN/DRWeibo/base_bigcn_directed_valloss_layer3_manifest/seed_0/best_model.pth.m"
  ["EIN"]="experiments/EIN/DRWeibo/EIN_Bigcn_directed_valloss_layer3_manifest/seed_0/best_model.pth.m"
)

declare -A EARLY_TEST_BASES=(
  ["our_model"]="dataset/DRWeibo/early_detection/resgcn/seed_0"
  ["see"]="dataset/DRWeibo/early_detection/resgcn/seed_0"
  ["ragcl"]="dataset/DRWeibo/early_detection/resgcn/seed_0"
  ["bigcn"]="dataset/DRWeibo/early_detection/plain_bigcn/seed_0"
  ["EIN"]="dataset/DRWeibo/early_detection/bigcn/seed_0"
)

if [[ ! -f "${PROJECT_ROOT}/main.py" ]]; then
  echo "Error: main.py was not found under PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}" || exit 2

failures=()
run_count=0
total_runs=$(( ${#MODEL_NAMES[@]} * ${#TIME_POINTS[@]} ))

for model_name in "${MODEL_NAMES[@]}"; do
  config_path="${PROJECT_ROOT}/${CONFIG_PATHS[$model_name]}"
  checkpoint_path="${PROJECT_ROOT}/${CHECKPOINT_PATHS[$model_name]}"
  early_test_base="${PROJECT_ROOT}/${EARLY_TEST_BASES[$model_name]}"
  model_slug="${MODEL_SLUGS[$model_name]}"

  for time_point in "${TIME_POINTS[@]}"; do
    run_count=$((run_count + 1))
    early_test_root="${early_test_base}/${time_point}"
    log_file="${LOG_ROOT}/${model_slug}_${time_point}.log"

    echo
    echo "[${run_count}/${total_runs}] Model: ${model_name} | Time: ${time_point}"
    echo "Log: ${log_file}"

    "${PYTHON_BIN}" main.py \
      --config_filename "${config_path}" \
      --device "${DEVICE}" \
      --eval_only \
      --seed "${SEED}" \
      --checkpoint_path "${checkpoint_path}" \
      --early_test_root "${early_test_root}" \
      2>&1 | tee "${log_file}"

    command_status=${PIPESTATUS[0]}
    if (( command_status != 0 )); then
      failures+=("${model_name}/${time_point} (exit ${command_status})")
      echo "FAILED: ${model_name}/${time_point}" >&2
    else
      echo "DONE: ${model_name}/${time_point}"
    fi
  done
done

echo
echo "Evaluation finished. Logs: ${LOG_ROOT}"

if (( ${#failures[@]} > 0 )); then
  echo "Failed runs (${#failures[@]}):" >&2
  printf '  - %s\n' "${failures[@]}" >&2
  exit 1
fi

echo "All ${total_runs} runs completed successfully."
