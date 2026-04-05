#!/bin/bash
# Download MNIST-FC models from VNN-COMP 2022 benchmark.
# These are small fully-connected networks perfect for TP testing.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
mkdir -p "$MODELS_DIR"

BASE_URL="https://github.com/ChristopherBrix/vnncomp2022_benchmarks/raw/main/benchmarks/mnist_fc"

echo "Downloading MNIST-FC models from VNN-COMP 2022..."

for model in mnist-net_256x2 mnist-net_256x4 mnist-net_256x6; do
    DEST="$MODELS_DIR/${model}.onnx"
    if [ -f "$DEST" ]; then
        echo "  $model.onnx already exists, skipping."
    else
        echo "  Downloading $model.onnx ..."
        curl -fSL "$BASE_URL/onnx/${model}.onnx" -o "$DEST"
        echo "  Done."
    fi
done

# Also download a vnnlib instance for reference
VNNLIB_DIR="$SCRIPT_DIR/vnnlib"
mkdir -p "$VNNLIB_DIR"

INSTANCES_URL="$BASE_URL/instances.csv"
INSTANCES_FILE="$VNNLIB_DIR/instances.csv"
if [ ! -f "$INSTANCES_FILE" ]; then
    echo "Downloading instances.csv..."
    curl -fSL "$INSTANCES_URL" -o "$INSTANCES_FILE"
fi

# Download first vnnlib spec for 256x2
FIRST_SPEC=$(head -2 "$INSTANCES_FILE" | tail -1 | cut -d',' -f2)
if [ -n "$FIRST_SPEC" ]; then
    SPEC_NAME="$(basename "$FIRST_SPEC")"
    SPEC_DEST="$VNNLIB_DIR/$SPEC_NAME"
    if [ ! -f "$SPEC_DEST" ]; then
        echo "Downloading sample vnnlib: $SPEC_NAME ..."
        curl -fSL "$BASE_URL/$FIRST_SPEC" -o "$SPEC_DEST" || echo "  (failed, not critical)"
    fi
fi

echo ""
echo "Models downloaded to: $MODELS_DIR"
ls -la "$MODELS_DIR"
