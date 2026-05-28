#!/usr/bin/env bash
set -euo pipefail

group="${1:-single_left}"
alg_name="${2:-dp3}"
seed="${3:-0}"
gpu_id="${4:-0}"

case "${group}" in
  single_left)
    task_name="ultradexgrasp_single_left"
    ;;
  single_right)
    task_name="ultradexgrasp_single_right"
    ;;
  bimanual)
    task_name="ultradexgrasp_bimanual"
    ;;
  *)
    echo "Unsupported group: ${group}" >&2
    exit 1
    ;;
esac

bash scripts/train_policy.sh "${alg_name}" "${task_name}" ultradex "${seed}" "${gpu_id}"
