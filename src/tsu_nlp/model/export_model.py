import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TracedT5(torch.nn.Module):
    def __init__(self, t5_model):
        super().__init__()
        self.t5 = t5_model
        
    def forward(self, input_ids, attention_mask):
        # For tracing, we use a fixed decoder input
        batch_size = input_ids.shape[0]
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=False
        )
        return outputs[0]  # Return logits only

def export_model(model_path, output_dir):
    """
    Export the T5 model to TorchScript format for Triton deployment.
    
    Args:
        model_path: Path to the HuggingFace model
        output_dir: Directory to save the exported model
    """
    try:
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        # Set model to evaluation mode
        model.eval()
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        # Wrap model for tracing
        wrapped_model = TracedT5(model)
        wrapped_model.eval()
        
        # Create example inputs for tracing
        example_text = "пример текста для нормализации"
        logger.info("Tokenizing example text")
        inputs = tokenizer(example_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Trace the model
        logger.info("Tracing model with TorchScript")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapped_model,
                (input_ids, attention_mask),
                strict=False
            )
        
        # Save the traced model
        os.makedirs(output_dir, exist_ok=True)
        traced_model_path = os.path.join(output_dir, "model.pt")
        logger.info(f"Saving traced model to {traced_model_path}")
        torch.jit.save(traced_model, traced_model_path)
        
        # Save tokenizer config
        logger.info("Saving tokenizer configuration")
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model successfully exported to: {traced_model_path}")
        logger.info("Tokenizer config saved in the same directory")
        
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise

if __name__ == "__main__":
    model_path = "models_cache/models--saarus72--russian_text_normalizer/snapshots/9ddc3152c12c46000d78d74c4325d8c5e2486f0b"
    output_dir = "model_repository/text_normalization/1"
    export_model(model_path, output_dir) 