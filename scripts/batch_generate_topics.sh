#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/AutoSurvey
PY=/root/miniconda3/envs/autosurvey/bin/python
TOPIC_FILE="${1:-$ROOT/topic.txt}"
OUT_ROOT="${2:-$ROOT/outputs/topic_bundle}"
MODEL_KEY="${MODEL_KEY:-}"
IMAGE_KEY="${IMAGE_KEY:-}"
MODEL_NAME="${MODEL_NAME:-gpt-4o}"
MODEL_URL="${MODEL_URL:-https://yunwu.ai/v1/chat/completions}"
IMAGE_MODEL="${IMAGE_MODEL:-gpt-image-2-all}"
IMAGE_URL="${IMAGE_URL:-https://api.openai.com/v1/images/generations}"
EMBED_MODEL="${EMBED_MODEL:-$ROOT/model/nomic-embed-text-v1}"
DB_PATH="${DB_PATH:-$ROOT/database}"
LOG_PATH="$OUT_ROOT/batch.log"
MANIFEST_PATH="$OUT_ROOT/manifest.tsv"

if [[ -z "$MODEL_KEY" ]]; then
  echo "MODEL_KEY is required" >&2
  exit 1
fi

if [[ -z "$IMAGE_KEY" ]]; then
  echo "IMAGE_KEY is required" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
cp "$TOPIC_FILE" "$OUT_ROOT/topic.txt"

"$PY" - <<'PY' "$TOPIC_FILE" "$MANIFEST_PATH"
from pathlib import Path
import re
import sys

topic_file = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
topics = [line.strip() for line in topic_file.read_text(encoding="utf-8").splitlines() if line.strip()]

seen = {}
rows = []
for idx, topic in enumerate(topics, 1):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", topic).strip("_").lower()
    slug = slug[:96] or f"topic_{idx:02d}"
    count = seen.get(slug, 0) + 1
    seen[slug] = count
    if count > 1:
        slug = f"{slug}_{count}"
    rows.append(f"{idx:02d}\t{slug}\t{topic}")

manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
PY

printf '[%s] Batch generation started\n' "$(date -u '+%F %T UTC')" | tee -a "$LOG_PATH"

while IFS=$'\t' read -r idx slug topic; do
  dest="$OUT_ROOT/${idx}_${slug}"
  mkdir -p "$dest"
  printf '\n[%s] Topic %s: %s\n' "$(date -u '+%F %T UTC')" "$idx" "$topic" | tee -a "$LOG_PATH"

  if [[ "$topic" == "In-context Learning" ]]; then
    rm -rf "$dest"
    cp -a "$ROOT/outputs/autosurvey" "$dest"
    printf '[%s] Reused existing output for %s\n' "$(date -u '+%F %T UTC')" "$topic" | tee -a "$LOG_PATH"
    continue
  fi

  if compgen -G "$dest/*.pdf" > /dev/null && compgen -G "$dest/*.md" > /dev/null; then
    printf '[%s] Existing completed output found, skipping %s\n' "$(date -u '+%F %T UTC')" "$topic" | tee -a "$LOG_PATH"
    continue
  fi

  HF_ENDPOINT=https://hf-mirror.com \
  "$PY" "$ROOT/main.py" \
    --gpu 0 \
    --topic "$topic" \
    --saving_path "$dest" \
    --db_path "$DB_PATH" \
    --embedding_model "$EMBED_MODEL" \
    --model "$MODEL_NAME" \
    --api_url "$MODEL_URL" \
    --api_key "$MODEL_KEY" \
    --image_model "$IMAGE_MODEL" \
    --image_api_key "$IMAGE_KEY" \
    --image_api_url "$IMAGE_URL" 2>&1 | tee -a "$LOG_PATH"

  "$PY" "$ROOT/evaluation.py" \
    --gpu 0 \
    --saving_path "$dest" \
    --topic "$topic" \
    --model "$MODEL_NAME" \
    --api_url "$MODEL_URL" \
    --api_key "$MODEL_KEY" \
    --db_path "$DB_PATH" \
    --embedding_model "$EMBED_MODEL" 2>&1 | tee -a "$LOG_PATH"
done < "$MANIFEST_PATH"

printf '\n[%s] Batch generation finished\n' "$(date -u '+%F %T UTC')" | tee -a "$LOG_PATH"
