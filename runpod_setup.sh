#!/bin/bash

# RunPod Setup Script for Coordinate Transformer Training
# This script sets up the environment and starts training on RunPod

echo "=== Coordinate Transformer RunPod Setup ==="
echo "Starting setup at $(date)"

# Update system packages
apt-get update
apt-get install -y git wget

# Clone the repository
echo "Cloning repository..."
cd /workspace
git clone https://github.com/kanadm12/Coordinate-Transformer-.git
cd Coordinate-Transformer-

# Create necessary directories
mkdir -p checkpoints_coord_transformer
mkdir -p logs

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy tqdm pillow matplotlib tensorboard nibabel

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Display training configuration
echo "=== Training Configuration ==="
cat config_coord_transformer.json

echo ""
echo "=== Setup Complete ==="
echo "To start training, run:"
echo "  python train_coord_transformer_4gpu.py --config config_coord_transformer.json"
echo ""
echo "For single GPU training:"
echo "  python train_coord_transformer_4gpu.py --config config_coord_transformer.json --num_gpus 1"
echo ""
echo "For distributed training with 4 GPUs:"
echo "  python train_coord_transformer_4gpu.py --config config_coord_transformer.json --num_gpus 4"
