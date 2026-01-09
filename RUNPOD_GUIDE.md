# RunPod Training Guide for Coordinate Transformer

## Quick Start on RunPod

### 1. Create a RunPod Instance

1. Go to [RunPod.io](https://runpod.io) and sign in
2. Click "Deploy" and select a GPU pod (recommended: 4x A40 or 4x RTX 4090)
3. Select PyTorch template or use a base CUDA image
4. Deploy the pod

### 2. Setup Environment

Once your pod is running, open a terminal and run:

```bash
# Download and run the setup script
wget https://raw.githubusercontent.com/kanadm12/Coordinate-Transformer-/main/runpod_setup.sh
chmod +x runpod_setup.sh
./runpod_setup.sh
```

Or manually:

```bash
cd /workspace
git clone https://github.com/kanadm12/Coordinate-Transformer-.git
cd Coordinate-Transformer-
pip install torch torchvision torchaudio numpy scipy tqdm pillow matplotlib tensorboard
```

### 3. Prepare Your Data

Upload your dataset to `/workspace/drr_patient_data` or update the path in `config_coord_transformer.json`:

```json
"data": {
  "dataset_path": "/workspace/drr_patient_data",
  ...
}
```

### 4. Start Training

**Single GPU:**
```bash
python train_coord_transformer_4gpu.py --config config_coord_transformer.json --num_gpus 1
```

**Multi-GPU (4 GPUs):**
```bash
python train_coord_transformer_4gpu.py --config config_coord_transformer.json --num_gpus 4
```

### 5. Monitor Training

Training logs and checkpoints will be saved to:
- Logs: `logs/`
- Checkpoints: `checkpoints_coord_transformer/`

You can monitor training with TensorBoard:
```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

Then access it via RunPod's port forwarding.

## Configuration

Edit `config_coord_transformer.json` to customize:

- **Model size**: `coord_embed_dim`, `num_transformer_blocks`, `num_heads`
- **Training**: `batch_size`, `learning_rate`, `num_epochs`
- **Data**: `dataset_path`, `max_patients`, `train_split`

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` in config (try 1 per GPU)
- Reduce `volume_size` (e.g., [32, 32, 32])
- Reduce `coord_embed_dim` (e.g., 256)

**Slow training?**
- Increase `num_workers` in config
- Use fewer transformer blocks initially
- Enable mixed precision (already enabled in script)

**Dataset issues?**
- Ensure dataset path is correct
- Check data format matches PatientDRRDataset expectations
- Verify sufficient storage space

## Saving Your Work

Before stopping your pod:

```bash
# Save checkpoints to external storage
# Option 1: Push to cloud storage
# Option 2: Download via RunPod interface
# Option 3: Sync to GitHub

git add checkpoints_coord_transformer/
git commit -m "Training checkpoints"
git push
```

## Cost Optimization

- Start with 1 GPU for testing
- Use Spot instances for long training runs
- Monitor and stop when validation loss plateaus
- Save checkpoints regularly in case of interruption
