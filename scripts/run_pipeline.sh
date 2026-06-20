#!/usr/bin/env bash
# run_pipeline.sh
# ----------------
# Runs Stages 1-3 end to end with sensible defaults. Edit the variables below
# (or override via environment variables) to change language, corpus size,
# and tokenizer hyperparameters.
#
# Usage:
#   chmod +x scripts/run_pipeline.sh
#   ./scripts/run_pipeline.sh

set -euo pipefail

LANG_CODE="${LANG_CODE:-en}"
SIZE_LIMIT_MB="${SIZE_LIMIT_MB:-100}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
MIN_FREQUENCY="${MIN_FREQUENCY:-2}"
USE_FAST_EXTRACTION="${USE_FAST_EXTRACTION:-true}"   # true = HF datasets path, false = raw XML dump

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIR="$ROOT_DIR/corpus"
TOKENIZER_OUT="$ROOT_DIR/tokenizer_output"

echo "=== Stage 1: Data extraction ($LANG_CODE wikipedia, ${SIZE_LIMIT_MB}MB) ==="
if [ "$USE_FAST_EXTRACTION" = "true" ]; then
    python "$ROOT_DIR/data_extraction/extract_wikipedia_fast.py" \
        --lang "$LANG_CODE" --size-limit-mb "$SIZE_LIMIT_MB" --out-dir "$CORPUS_DIR"
else
    python "$ROOT_DIR/data_extraction/extract_wikipedia.py" \
        --lang "$LANG_CODE" --size-limit-mb "$SIZE_LIMIT_MB" --out-dir "$CORPUS_DIR"
fi

echo "=== Stage 2: BPE training (vocab_size=$VOCAB_SIZE, min_frequency=$MIN_FREQUENCY) ==="
python "$ROOT_DIR/tokenizer_training/train_bpe.py" \
    --corpus "$CORPUS_DIR/${LANG_CODE}_wikipedia_corpus.txt" \
    --vocab-size "$VOCAB_SIZE" \
    --min-frequency "$MIN_FREQUENCY" \
    --out-dir "$TOKENIZER_OUT"

echo "=== Stage 3: Evaluation ==="
python "$ROOT_DIR/evaluation/evaluate.py" \
    --tokenizer-path "$TOKENIZER_OUT/tokenizer.json" \
    --benchmarks wikitext-103 ptb \
    --out "$TOKENIZER_OUT/evaluation_report.json"

echo ""
echo "Pipeline complete. Tokenizer + evaluation report are in: $TOKENIZER_OUT"
echo "Next: start the API (uvicorn api.main:app --reload) and the frontend (npm run dev)."
