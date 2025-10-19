#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH=${1:-"${ROOT_DIR}/experiments/configs/gmm_em.yaml"}

python "${ROOT_DIR}/main.py" --config "${CONFIG_PATH}"
