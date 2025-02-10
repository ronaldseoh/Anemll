#!/bin/bash

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

print_usage() {
    echo "Usage: $0 --input <converted_model_dir> [--output <output_dir>] [--org <huggingface_org>]"
    echo "Options:"
    echo "  --input    Directory containing converted model files (required)"
    echo "  --output   Output directory for HF distribution (optional, defaults to input_dir/hf_dist)"
    echo "  --org      Hugging Face organization/account (optional, defaults to anemll)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --org)
            HF_ORG="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory is required"
    print_usage
fi

# Convert input path to absolute path
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)" || {
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
}

# Read meta.yaml
if [ ! -f "$INPUT_DIR/meta.yaml" ]; then
    echo "Error: meta.yaml not found in input directory"
    exit 1
fi

# Extract model name and parameters from meta.yaml
MODEL_NAME=$(grep "name:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f4)
MODEL_VERSION=$(grep "version:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f4)
CONTEXT_LENGTH=$(grep "context_length:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
BATCH_SIZE=$(grep "batch_size:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
MODEL_PREFIX=$(grep "model_prefix:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
NUM_CHUNKS=$(grep "num_chunks:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
LUT_FFN=$(grep "lut_ffn:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)
LUT_LMHEAD=$(grep "lut_lmhead:" "$INPUT_DIR/meta.yaml" | cut -d' ' -f6)

# Construct full model name with version
FULL_MODEL_NAME="${MODEL_NAME}_${MODEL_VERSION}"

# Set output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$INPUT_DIR/hf_dist"
fi

# Set default HF organization if not specified
if [ -z "$HF_ORG" ]; then
    HF_ORG="anemll"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Preparing distribution for $FULL_MODEL_NAME..."

# Function to compress mlmodelc directory
compress_mlmodelc() {
    local dir=$1
    local base=$(basename "$dir" .mlmodelc)
    echo "Compressing $base.mlmodelc..."
    (cd "$INPUT_DIR" && zip -r "$OUTPUT_DIR/${base}.mlmodelc.zip" "${base}.mlmodelc")
}

# Compress all mlmodelc files
compress_mlmodelc "${MODEL_PREFIX}_embeddings"
compress_mlmodelc "${MODEL_PREFIX}_lm_head_lut${LUT_LMHEAD}"
for ((i=1; i<=NUM_CHUNKS; i++)); do
    chunk_num=$(printf "%02d" $i)
    compress_mlmodelc "${MODEL_PREFIX}_FFN_PF_lut${LUT_FFN}_chunk_${chunk_num}of$(printf "%02d" $NUM_CHUNKS)"
done

# Copy required files
cp "$INPUT_DIR/meta.yaml" "$OUTPUT_DIR/"
cp "$INPUT_DIR/tokenizer.json" "$OUTPUT_DIR/"
cp "$INPUT_DIR/tokenizer_config.json" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/tests/chat.py" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/tests/chat_full.py" "$OUTPUT_DIR/"

# Create README.md from template
README_TEMPLATE="$SCRIPT_DIR/readme.template"
if [ ! -f "$README_TEMPLATE" ]; then
    echo "Error: readme.template not found at $README_TEMPLATE"
    exit 1
fi

# Read template and replace placeholders
sed -e "s|%NAME_OF_THE_FOLDER_WE_UPLOAD%|$FULL_MODEL_NAME|g" \
    -e "s|%PATH_TO_META_YAML%|./meta.yaml|g" \
    -e "s|%HF_ORG%|$HF_ORG|g" \
    "$README_TEMPLATE" > "$OUTPUT_DIR/README.md"

echo "Distribution prepared in: $OUTPUT_DIR"
echo
echo "To upload to Hugging Face, use:"
echo "huggingface-cli upload $HF_ORG/$FULL_MODEL_NAME $OUTPUT_DIR" 