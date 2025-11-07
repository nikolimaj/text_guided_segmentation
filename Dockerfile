# Use official Python 3.8 slim image
FROM python:3.8-slim

# Set environment variables to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \

    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install torch==1.12.1+cpu torchvision==0.13.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install pytorch_lightning==1.9.0 torchmetrics==0.10.3 \
    transformers==4.24.0 monai==1.0.1 pandas einops

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Default command
CMD ["python"]
