#!/bin/bash
# Quick start script for training YOLOv8 License Plate Detector

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║   YOLOv8 License Plate Detector Training Pipeline                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

# Activate virtual environment
echo "Activating virtual environment..."
source train_env/bin/activate

# Run training
echo ""
echo "Starting training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python training/train.py

# After training completes
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate model:"
    echo "   python training/evaluate.py"
    echo ""
    echo "2. Test on images:"
    echo "   python training/predict.py --source dataset/test/images"
    echo ""
    echo "3. Export model:"
    echo "   python training/export.py --format all --include-guide"
    echo ""
    echo "4. Generate report:"
    echo "   python training/generate_report.py"
else
    echo ""
    echo "✗ Training failed. Check logs for details:"
    echo "   tail -f models/plate_detector/train.log"
fi
