#!/usr/bin/env bash
# Puts every project code folder on PYTHONPATH, then execs python3 with the
# given arguments. Usage: ./run.sh dag_codes/dag_main.py [args...]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CODE_DIRS=(
  "dag_codes"
  "tree_codes"
)

PYPATH="${PYTHONPATH:-}"
for d in "${CODE_DIRS[@]}"; do
  PYPATH="${PROJECT_ROOT}/${d}:${PYPATH}"
done

export PYTHONPATH="$PYPATH"
exec python3 "$@"
