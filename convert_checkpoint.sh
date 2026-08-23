#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./convert_checkpoint.sh <checkpoints_dir>

Examples:
  ./convert_checkpoint.sh /path/to/checkpoints
EOF
}

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BASE="$1"

if [[ ! -d "$BASE" ]]; then
  echo "Base directory does not exist: $BASE" >&2
  exit 1
fi

for model_dir in "$BASE"/iter_*/model; do
  if [ ! -d "$model_dir" ]; then
    continue
  fi
  iter_dir=$(dirname "$model_dir")
  out="$iter_dir/model.pt"

  if [ -f "$out" ]; then
    echo "Skipping $iter_dir (already has model.pt)"
    continue
  fi

  echo "Converting $iter_dir"
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  python "$SCRIPT_DIR/convert_distcp.py" "$model_dir" "$iter_dir"
done