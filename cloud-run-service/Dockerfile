# Use NVIDIA CUDA base image for GPU support (required for pyannote.audio)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# OR use this for CPU-only testing (not recommended for production):
# FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "main.py"]
