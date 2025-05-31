import torch

print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("No CUDA available. Make sure you have:")
    print("1. NVIDIA GPU")
    print("2. NVIDIA drivers installed")
    print("3. CUDA toolkit installed")
    print("4. PyTorch with CUDA support installed")
    
print(f"\nPyTorch version: {torch.__version__}")
print(f"PyTorch CUDA version it was built with: {torch.version.cuda}") 