FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git ffmpeg wget curl \
                                           && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspace

# Copy repo
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command
CMD ["python3", "generate.py"]
