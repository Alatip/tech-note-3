#!/bin/sh
# Chen 2026 outcome matrices, HF josefchen/co-failure-67-models (CC-BY 4.0), snapshot f2dd304 (2026-09-25)
cd "$(dirname "$0")"
for f in matrix_marketE2_final.json matrix_marketMH_final.json; do
  [ -f "$f" ] || curl -sL -o "$f" "https://huggingface.co/datasets/josefchen/co-failure-67-models/resolve/main/runs/$f"
done
ls -la *.json
