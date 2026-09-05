#!/usr/bin/env bash

# Run early-detection evaluation for all Pheme models and time points.
#
# Optional environment variables:
#   PROJECT_ROOT  Project directory (default: /public/wc/EIN-main)
#   PYTHON_BIN    Python executable (default: python)
#   LOG_ROOT      Output directory for per-run logs

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/wc/EIN-main}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED=3
TIME_POINTS=(0m 10m 20m 30m 40m 50m 60m 120m)
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/pheme_early_eval/$(date +%Y%m%d_%H%M%S)}"

MODEL_NAMES=(
  "Our model"
  "EIN"
  "BiGCN"
  "RAGCL"
  "SEE"
  "resgcn"
)

declare -A MODEL_SLUGS=(
  ["Our model"]="our_model"
  ["EIN"]="ein"
  ["BiGCN"]="bigcn"
  ["RAGCL"]="ragcl"
  ["SEE"]="see"
  ["resgcn"]="resgcn"
)

declare -A CONFIG_PATHS=(
  ["Our model"]="configs/EIN/Pheme_BiGCN_UncertaintySemanticChange.yaml"
  ["EIN"]="configs/EIN/Pheme.yaml"
  ["BiGCN"]="configs/EIN/Pheme_RAGCL_BiGCN_word2vec.yaml"
  ["RAGCL"]="configs/EIN/Pheme_RAGCL_BiGCN_word2vec.yaml"
  ["SEE"]="configs/EIN/Pheme_SEEGraphMAE.yaml"
  ["resgcn"]="configs/EIN/Pheme_RAGCL_ResGCN_word2vec.yaml"
)

declare -A CHECKPOINT_PATHS=(
  ["Our model"]="experiments/EIN/Pheme/test_bigcn_duibixuexi_aux1.0_mlp_nobias/seed_3/best_model.pth.m"
  ["EIN"]="experiments/EIN/Pheme/EIN_bigcn_directed_valloss_manifest/seed_3/best_model.pth.m"
  ["BiGCN"]="experiments/EIN/Pheme/base_bigcn_directed_valloss_manifest/seed_3/best_model.pth.m"
  ["RAGCL"]="experiments/EIN/Pheme/ragcl_bigcn_directed_valloss_manifest/seed_3/best_model.pth.m"
  ["SEE"]="experiments/EIN/Pheme/see_graphmae_directed_valloss_word2vec_manifest1/seed_3/best_model.pth.m"
  ["resgcn"]="experiments/EIN/Pheme/base_resgcn_directed_valloss_layer2_manifest/seed_3/best_model.pth.m"
)

declare -A EARLY_TEST_BASES=(
  ["Our model"]="dataset/Pheme/early/bigcn_kind/seed_3"
  ["EIN"]="dataset/Pheme/early/bigcn_kind/seed_3"
  ["BiGCN"]="dataset/Pheme/early/bigcn_kind/seed_3"
  ["RAGCL"]="dataset/Pheme/early/ragcl_central/seed_3"
  ["SEE"]="dataset/Pheme/early/resgcn_and_othermodel"
  ["resgcn"]="dataset/Pheme/early/resgcn_and_othermodel"
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
