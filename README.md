# ANEMLL

ANEMLL (pronounced like "animal") is an open-source project focused on accelerating the porting of Large Language Models (LLMs) to tensor processors, starting with the Apple Neural Engine (ANE).

## Goals
> The goal is to provide a fully open-source pipeline from model conversion to inference for common LLM architectures running on ANE.
> This enables seamless integration and on-device inference for low-power applications on edge devices, ensuring maximum privacy and security.
> This is critical for autonomous applications, where models run directly on the device without requiring an internet connection.

We aim to:
- Provide flexible and easy to use library/framework to port LLMs to ANE directly from Hugging Face models
- Provide on-device examples for iOS and macOS swift or C/C++ Applications


> [!Important]
> This is Alpha Release 0.1.0 for the library. It is designed to process Model Weights directly from Hugging Face models and convert them to the CoreML format for Apple Neural Engine (ANE for short).
> - This release only supports LLAMA models including DeepSeek distilled models on LLaMA 3.1 architecture
> - The future release will add support for more models and architectures

## Basic Workflow

See [Model Conversion Guide](./docs/convert.md) and [DeepSeek Model Conversion Guide](./docs/ConvertingDeepSeek.md) for more details.

1. Download the model from Hugging Face
2. Convert the model to the CoreML format using ANEMLL
3. Run the model on the Apple Neural Engine using provided example code `chat.py`

### Conversion Process Overview
- ANE models on iOS are limited to 1GB file size. macOS will work with ~2GB
- We split models during the conversion process to avoid this limit

### Model Components
There are 3 parts for LLM:
1. Embeddings
2. Feed Forward Network/layers 
3. LM Head

> LLaMA Model ANE optimized implementation is in `./anemall/models/llama_model.py`

For FFN, we can split it into multiple chunks to allow for big models (like 8GB LLaMA/DeepSeek)

### Conversion Steps

1. **ANE_converter**:
   - `./anemall/ANE_converter.py` creates MLPackages for each part
   - We also create "Prefill" models for KV cache
   - This implementation uses Stateful API for ANE, introduced in iOS 18 / macOS 15

2. **Combine Models**:
   - After creating MLPackages, we merge FFN and prefill chunks into Multi-Function Chunks
   - This reduces weight size by 50% as KV pre-fill and FFN use the same weights
   - Processed by `./anemall/combine_models.py`

3. **Compile Models**:
   - Convert to MLModelC format for on-device inference
   - Done via `./anemall/utils/compile_models.py`

See [Model Conversion Documentation](./docs/model_conversion.md) for more details.

## Testing

We provide two chat interfaces:
- `chat.py` - Basic chat interface for quick testing
- `chat_full.py` - Advanced chat with conversation history management

Features of chat_full.py:
- Maintains full conversation history within context window
- Automatically truncates older messages when needed
- Shifts context window dynamically during long responses
- Shows generation speed and token statistics
- Better handles multi-turn conversations

Example running chat_full.py:
```bash
python tests/chat_full.py \
  --embed ./llama_embeddings \
  --lmhead llama_lm_head_lut6 \
  --ffn llama_FFN_PF_lut6_chunk_01of02 \
  --tokenizer ../Meta-Llama-3.1-8B
```

## Installation

### System Requirements
- macOS Sequoia with Apple Neural Engine
- Minimum 16GB RAM
- Python 3.9

### Setup Steps

1. Install ANEMLL:
We recommend creating a new virtual environment for this project.
```bash
python -m venv anemll-env
source anemll-env/bin/activate
pip install -r requirements.txt
# pip install anemll
# due to Alpha Release, we do not recommend installing ANEMLL as a package yet
```



## Model Support

Currently optimized for:
- Meta's LLaMA 3.2 1B and 8B (1024 context) model including DeepSeek R1 8GB distilled model
- More models are coming soon

## Acknowledgements

### Core Technologies
- Thanks to [@apple](https://apple.com) for developing the Apple Neural Engine 
- Thanks to Appel CoreML Tools team for providing the tools https://github.com/apple/coremltools
- Thanks to [@huggingface](https://huggingface.co) for providing the transformers library and models

### Inspirations, feedback and other resources
- Stephen Panaro https://x.com/flat for feedback and coreml-llm-cli https://github.com/smpanaro/coreml-llm-cli 
- Seba https://x.com/CulStory for inspiration with fast ANE models. https://huggingface.co/seba
- Maynard Handley https://x.com/handleym99 For indepth ANE resources https://github.com/name99-org/AArch64-Explore/blob/main/vol7%20ANE.nb.pdf and feedback

## Contributing

> [!Note]
> We welcome contributions! Please read our contributing guidelines before submitting PRs.

Feel free to submit issues and pull requests to improve **ANEMLL**!

## License

ANEMLL is licensed under the MIT License.

## Links & Resources

- 🌐 Website: [anemll.com](https://anemll.com)
- 🤗 Models: [huggingface.co/anemll](https://huggingface.co/anemll)
- 📱 Twitter: [@anemll](https://x.com/anemll)
- 💻 GitHub: [github.com/anemll](https://github.com/anemll)

## Contact

For any questions or support, reach out to us at [realanemll@gmail.com](mailto:realanemll@gmail.com)
