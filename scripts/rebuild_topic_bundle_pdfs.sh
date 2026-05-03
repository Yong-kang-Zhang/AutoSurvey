#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/AutoSurvey
PY=/root/miniconda3/envs/autosurvey/bin/python
OUT_ROOT="${1:-$ROOT/outputs/topic_bundle}"

find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | while read -r topic_dir; do
  md_file="$(find "$topic_dir" -maxdepth 1 -type f -name '*.md' ! -name '_running_preview.md' | sort | head -n 1)"
  if [[ -z "${md_file:-}" ]]; then
    continue
  fi

  echo "[rebuild] $md_file"
  PYTHONPATH="$ROOT" "$PY" - <<'PY' "$md_file"
import sys
from src.latex_converter import MD2LatexConverter

md_path = sys.argv[1]
converter = MD2LatexConverter(md_path)
converter.convert()
PY
done
