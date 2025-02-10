# ANEMLL Convert Workflow

> [!Important]
> This guide explains how to convert and run LLAMA 3.2 1B model using ANEMLL.
> - ANE models on iOS are limited to 1GB file size
> - macOS supports up to ~2GB
> - Models are split during conversion to avoid these limits
> - Requires macOS 15 or later
> - This model is quantized to 6-bit LUT, its a bit slower vs 4-bit LUT, but provides better quality.
> - For the fastest performance, use 4-bit LUT and 512 context length!

# Quick Conversion Using Script

For convenience, you can use the provided conversion script to perform all steps in one go. For detailed information about batch conversion, see [Convert Model Script](convert_model.md).

```bash
./anemll/utils/convert_model.sh --model <path_to_model> --output <output_directory> [options]
```

Basic usage example:
```bash
./anemll/utils/convert_model.sh \
    --model ../Meta-Llama-3.2-1B \
    --output ./converted_models
```

> See [Convert Model Guide](convert_model.md) for detailed parameters and examples.

# Manual Conversion Steps

If you need more control over the conversion process, you can follow these detailed steps:

# This guide is for LLAMA 3.2 1B Conversion and testing with chat.py app

1. Download the model from Hugging Face:
URL: https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main   

2. Convert the model to the CoreML format using ANEMLL
3. Run the model on the Apple Neural Engine using provide example code chat.py

ANE models on iOS are limited to 1GB file size. macOS will work with ~2GB.
We split modeles during conversion process to avoid this limit.

Generally there are 3 models' parts for LLM:  embedding, Feed Forward Network/layers and LM Head.
We call it part: 1, 2 and 3 respectively.
LLama Model ANE optimized implemtation is in ./anemall/models/llama_model.py
For FFN, we can split it in multiple chunks to allow for big models (like 8GB LLama/DeepSeek)

# Components

## ANE_converter
./anemall/ANE_converter.py is the file that processes this by craeteing MLPakages for each part.
We also create "Prefill" models for KV cache.  
This implementation is using Stateful API for ANE, introduced in iOS 18 / macOS 15.

## Combine_models
./anemall/utils/combine_models.py is the file that combines FFN and prefill chunks into Multi-Function Chunks

## Compile_models
./anemll/utils/compile_models.py is the file that converts the model to MLModelC format for device inference

## chat.py
./tests/chat.py is the file that runs the model on the Apple Neural Engine using Python



Below is generic workflow to convert LLAma 3.1 with 1024 context length and prefill h64 batch size

## Conversion Steps

### 1. Convert Embeddings (Part 1)
> Creates `llama_embeddings.mlpackage`

```bash
python -m anemll.ane_converter.llama_converter \
    --part 1 \
    --model "../Meta-Llama-3.2-1B"
```

### 2. Convert LM Head (Part 3)
> Creates `llama_lm_head_lut6.mlpackage` with 6-bit LUT quantization
```bash
python -m anemll.ane_converter.llama_converter \
    --part 3 \
    --lut 6 \
    --model "../Meta-Llama-3.2-1B"
```

### 3. Convert FFN (Part 2)
> CConvert FFN part  (2) splitting it into 2 chunks and 6-bit LUT quantization, 
it creates `llama_FFN_PF_lut6_chunk_[01-02]of02.mlpackage`
```bash
python -m anemll.ane_converter.llama_converter \
    --part 2 \
    --lut 6 \
    --chunk 2 \
    --context-length 1024 \
    --batch-size 64 \
    --model "../Meta-Llama-3.2-1B"
```

### 4. Convert Prefill (Part 2_prefill)
> Creates prefill model chunks for KV cache optimization
```bash
python -m anemll.ane_converter.llama_converter \
    --part 2_prefill \
    --lut 6 \
    --chunk 2 \
    --context-length 1024 \
    --batch-size 64 \
    --model "../Meta-Llama-3.2-1B"
```


### 5. Combine Models
After we have MLpackages, we merge FFN and prefill chunks into Multi-Function Chunks.
This allows us to reduce weight size by 50% because KV pre-fill and FFN are using the same weights.

```bash
# Basic usage (in models directory)
python ./anemll/utils/combine_models.py \
    --lut 6 \
    --chunk 2

# With explicit input/output directories
python ./anemll/utils/combine_models.py \
    --lut 6 \
    --chunk 2 \
    --input ./models \
    --output ./combined_models
```

### 6. Compile Models
> Converts to MLModelC format for device inference
```bash
python ./anemll/utils/compile_models.py 1
python ./anemll/utils/compile_models.py 3 --lut 6
python ./anemll/utils/compile_models.py 2 --lut 6 --chunk 2
```

### 7. Test with chat.py
> There are two options for testing:

#### Standard Chat (chat.py)

 Option 1 - Using meta.yaml (recommended):
 ```bash
 python ./tests/chat.py \
     --meta ./converted_models/meta.yaml
 ```
 
 Option 2 - Manual configuration:
 ```bash
 python ./tests/chat.py \
     --embed llama_embeddings \
     --lmhead llama_lm_head_lut6 \
     --ffn llama_FFN_PF_lut6_chunk_01of02 \
     --tokenizer ../Meta-Llama-3.2-1B \
     --context-length 1024
 ```

> Note: If you used [convert_model.sh](convert_model.md) to convert the model, you can also run chat using the generated meta.yaml:
```
## Output Files

After conversion and compilation, you should have:

### MLPackage Files (Intermediate)
- `llama_embeddings.mlpackage`
- `llama_lm_head_lut6.mlpackage`
- Two FFN chunk files:
  - `llama_FFN_PF_lut6_chunk_01of02.mlpackage`
  - `llama_FFN_PF_lut6_chunk_02of02.mlpackage`

### MLModelC Files (Final)
> These are the compiled files used for inference
- `llama_embeddings.mlmodelc/`
- `llama_lm_head_lut6.mlmodelc/`
- Two FFN chunk directories:
  - `llama_FFN_PF_lut6_chunk_01of02.mlmodelc/`
  - `llama_FFN_PF_lut6_chunk_02of02.mlmodelc/`

> [!Note]
> The .mlmodelc files are actually directories containing the compiled model assets.
> These are the files that should be included in your application for inference.

## Additional Information

- The FFN chunks are combined with prefill functionality to optimize storage
- Context length can be configured (example uses 1024)
- Batch size affects prefill performance (example uses 64)
- LUT quantization helps reduce model size (example uses 6-bit)

For more details about the implementation, see:
- `./anemall/models/llama_model.py` - Core model implementation
- `./anemall/ane_converter/llama_converter.py` - Conversion logic
- `./tests/chat.py` - Example chat interface

## See Also
- [Converting DeepSeek Models](ConvertingDeepSeek.md) - For larger model conversion
- [Compile Models Documentation](compile_models.md) - Details about compilation
- [Combine Models Documentation](combine_models.md) - Details about model combining

    

