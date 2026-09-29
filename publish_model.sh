#!/bin/sh
# Publish the finetuned LoRA adapter + matura.lol branded card to
# matura-lol/GLiNER2.5-multi-Decide-finetune (HF org) via a Pull Request.
# Run AFTER training completes (output/best holds the best adapter).
set -e

ADAPTER_DIR="/home/dj/Documents/ml.matugen/output/best"
CARD="/home/dj/Documents/ml.matugen/hf_model_card/README.md"
LOG="/home/dj/Documents/ml.matugen/hf_publish_model.log"

# Stage the files to upload: adapter + branded card
STAGE="/home/dj/Documents/ml.matugen/publish_stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$ADAPTER_DIR/adapter_config.json" "$ADAPTER_DIR/adapter_model.safetensors" "$STAGE/"
cp "$CARD" "$STAGE/README.md"

echo "Uploading to matura-lol/GLiNER2.5-multi-Decide-finetune via PR..."
hf upload matura-lol/GLiNER2.5-multi-Decide-finetune "$STAGE" \
  --type model \
  --revision refs/pr/publish \
  --commit-message "Add matura.lol LoRA finetune + branded model card" \
  --create-pr \
  > "$LOG" 2>&1 || { echo "FAILED — see $LOG"; tail -20 "$LOG"; exit 1; }
echo "OK. PR created — merge it at https://huggingface.co/models/matura-lol/GLiNER2.5-multi-Decide-finetune/discussions"
tail -5 "$LOG"