# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker cache
COPY pyproject.toml poetry.lock ./
COPY src/ ./src/
COPY data/ ./data/

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

# Copy only the requirements first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy the source code
COPY src/ ./src/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/dictionary data/results data/inputs models_cache

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV CUDA_VISIBLE_DEVICES=0

# Default command to run the normalizer in neural mode
ENTRYPOINT ["python", "-m", "src.tsu_nlp.model.model"]
CMD ["normalize", "neural"] 