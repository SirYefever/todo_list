# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (official recommended way)
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy only the requirements files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Create necessary directories for volume mounting
RUN mkdir -p /app/data/dictionary /app/data/results /app/data/inputs \
    /app/models_cache \
    /app/src

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV CUDA_VISIBLE_DEVICES=0

# Default command to run the normalizer in neural mode
ENTRYPOINT ["python", "-m", "src.tsu_nlp.model.model"]
CMD ["normalize", "neural"]

# Usage instructions for volume mounting:
# Run the container with the following volumes:
# docker run -v /path/to/local/data:/app/data \
#           -v /path/to/local/models_cache:/app/models_cache \
#           -v /path/to/local/src:/app/src \
#           your-image-name 