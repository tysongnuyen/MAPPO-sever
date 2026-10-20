#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${BASELINE_PYTHON:-python}" -u "$script_dir/train_greedy_mappo.py" "$@"
