import shutil
import os

# Create directories if they don't exist
os.makedirs('model_repository/text_normalization_encoder/1', exist_ok=True)
os.makedirs('model_repository/text_normalization_decoder/1', exist_ok=True)

# Copy encoder model
shutil.copy2(
    'model_repository/text_normalization/1/encoder_model.onnx',
    'model_repository/text_normalization_encoder/1/model.onnx'
)

# Copy decoder model
shutil.copy2(
    'model_repository/text_normalization/1/model.onnx',
    'model_repository/text_normalization_decoder/1/model.onnx'
)

# Copy tokenizer files to both directories
tokenizer_files = [
    'tokenizer.json',
    'merges.txt',
    'vocab.json',
    'added_tokens.json',
    'special_tokens_map.json',
    'tokenizer_config.json'
]

for file in tokenizer_files:
    src = f'model_repository/text_normalization/1/{file}'
    if os.path.exists(src):
        shutil.copy2(src, f'model_repository/text_normalization_encoder/1/{file}')
        shutil.copy2(src, f'model_repository/text_normalization_decoder/1/{file}')

print("Model repository setup completed successfully!") 