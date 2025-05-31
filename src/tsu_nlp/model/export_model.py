import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from optimum.exporters.onnx import main_export
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_model(model_path, output_dir):
    """
    Export the T5 model to ONNX format for Triton deployment.
    
    Args:
        model_path: Path to the HuggingFace model
        output_dir: Directory to save the exported model
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export model to ONNX
        logger.info(f"Exporting model from {model_path} to ONNX format")
        
        # Export the model using main_export
        main_export(
            model_name_or_path=model_path,
            output=output_dir,
            task="text2text-generation",
            opset=15,
            device="cuda" if torch.cuda.is_available() else "cpu",
            fp16=True
        )
        
        # Rename model files to match Triton requirements
        if os.path.exists(os.path.join(output_dir, "decoder_model.onnx")):
            os.rename(
                os.path.join(output_dir, "decoder_model.onnx"),
                os.path.join(output_dir, "model.onnx")
            )
        
        logger.info(f"Model successfully exported to ONNX format at: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise

if __name__ == "__main__":
    model_path = "models_cache/models--saarus72--russian_text_normalizer/snapshots/9ddc3152c12c46000d78d74c4325d8c5e2486f0b"
    output_dir = "model_repository/text_normalization/1"
    export_model(model_path, output_dir) 