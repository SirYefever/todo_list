FROM nvcr.io/nvidia/tritonserver:23.12-py3

WORKDIR /app

# Install ONNX Runtime
RUN pip install --no-cache-dir onnxruntime==1.16.3

# Copy model repository
COPY model_repository /models

# Run Triton server
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"] 