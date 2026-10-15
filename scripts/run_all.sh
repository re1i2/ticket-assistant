#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.prepare_data
python -m src.train --config configs/final.yaml
python -m src.train --config configs/distilbert.yaml
python -m src.run_experiments --tau 0.5
BEST=$(ls eval/predictions/finetuned_encoder.csv 2>/dev/null || ls eval/predictions/e5_lr.csv)
python -m eval.evaluate "$BEST" --slice set --slice dialect --tau 0.5
python -m eval.error_analysis "$BEST" --n 40
python scripts/make_samples.py
echo "Done"
