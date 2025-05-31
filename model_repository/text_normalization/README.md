# Text Normalization Model for Triton Inference Server

This directory contains the text normalization model prepared for deployment with NVIDIA Triton Inference Server.

## Model Information

- Model Type: T5 (T5ForConditionalGeneration)
- Task: Russian Text Normalization
- Input Format: Text tokens
- Output Format: Normalized text tokens

## Directory Structure

```
text_normalization/
├── config.pbtxt        # Triton model configuration
├── 1/                  # Model version directory
│   ├── model.pt       # TorchScript model
│   └── tokenizer files # T5 tokenizer configuration
└── README.md          # This documentation
```

## Model Configuration

The model is configured for:
- Maximum batch size: 8
- Dynamic batching enabled
- GPU acceleration with TensorRT FP16 optimization
- Variable sequence length support

## Input Tensors

1. `input_ids` (INT64):
   - Shape: [-1] (variable sequence length)
   - Contains tokenized input text

2. `attention_mask` (INT64):
   - Shape: [-1] (variable sequence length)
   - Contains attention mask for input tokens

## Output Tensor

1. `output_ids` (INT64):
   - Shape: [-1] (variable sequence length)
   - Contains generated normalized text tokens

## Pre/Post Processing

### Pre-processing:
1. Text tokenization using T5Tokenizer
2. Creation of attention masks
3. Input shape preparation

### Post-processing:
1. Token decoding using T5Tokenizer
2. Formatting output according to required CSV format: `id,after`

## Deployment Instructions

1. Install Triton Inference Server
2. Place this model repository in your Triton model store
3. Start Triton server:
   ```bash
   tritonserver --model-repository=/path/to/model_repository
   ```

## Performance Optimization

The model uses:
- TensorRT acceleration
- FP16 precision
- Dynamic batching
- GPU execution

## Input/Output Format Requirements

Input CSV format: `sentence_id,token_id,before`
Output CSV format: `id,after` where id is `sentence_id_token_id` 