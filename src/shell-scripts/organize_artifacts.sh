#!/usr/bin/env bash
# organize_artifacts.sh ---------------------------------------------------
# Organize all benchmark outputs into standard folders
# -------------------------------------------------------------------------

set -euo pipefail

echo "📁 Organizing benchmark artifacts..."

LOG_DIR="../../analysis/logs"
FIG_DIR="../../analysis/results/figures"
MODEL_CARD_DIR="../../analysis/results/model_cards"
METRICS_DIR="../../analysis/results/metrics"

# -------------------------------------------------------------------------
# Create folders if they don't exist
# -------------------------------------------------------------------------
mkdir -p "$LOG_DIR" "$FIG_DIR" "$MODEL_CARD_DIR" "$METRICS_DIR"

# -------------------------------------------------------------------------
# Move plots (F1, confusion, SHAP images)
# -------------------------------------------------------------------------
mv -v *_distribution*.png *_lineplot*.png "$FIG_DIR"/ 2>/dev/null || true

# -------------------------------------------------------------------------
# Move model cards and coefficient plots
# -------------------------------------------------------------------------
mv -v model_card_iter*.md "$MODEL_CARD_DIR"/ 2>/dev/null || true

echo "✅ Artifact organization complete!"
