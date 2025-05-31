FROM nvcr.io/nvidia/tritonserver:24.02-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime backend
RUN mkdir -p /opt/tritonserver/backends/onnxruntime && \
    wget -O /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so \
    https://github.com/triton-inference-server/onnxruntime_backend/releases/download/v2.43.0/libtriton_onnxruntime.so

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy model repository
COPY model_repository /models

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton
ENTRYPOINT ["tritonserver", "--model-repository=/models"] 