"""
Dataset for loading patient DRR and CT volume data
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json


class PatientDRRDataset(Dataset):
    """
    Dataset for loading patient CT volumes and corresponding DRR X-ray images.
    
    Expected data structure:
    dataset_path/
        patient_001/
            ct_volume.npy  # Shape: [D, H, W] or [C, D, H, W]
            xray_0.npy     # Shape: [H, W] or [C, H, W]
            xray_1.npy
            metadata.json  # Optional: projection parameters
        patient_002/
            ...
    """
    
    def __init__(self, data_path, target_volume_size=(64, 64, 64), 
                 max_patients=None, validate_alignment=False):
        """
        Args:
            data_path: Path to dataset directory
            target_volume_size: Target size for CT volumes (D, H, W)
            max_patients: Maximum number of patients to load (None for all)
            validate_alignment: Whether to validate data alignment
        """
        self.data_path = Path(data_path)
        self.target_volume_size = target_volume_size
        self.validate_alignment = validate_alignment
        
        # Find all patient directories
        self.patient_dirs = []
        if self.data_path.exists():
            self.patient_dirs = sorted([
                d for d in self.data_path.iterdir() 
                if d.is_dir() and d.name.startswith('patient_')
            ])
        
        if max_patients is not None:
            self.patient_dirs = self.patient_dirs[:max_patients]
        
        if len(self.patient_dirs) == 0:
            print(f"WARNING: No patient data found in {data_path}")
            print("Creating dummy dataset for testing...")
            # Create dummy data for testing if no real data exists
            self.use_dummy = True
            self.num_samples = max_patients if max_patients else 10
        else:
            self.use_dummy = False
            self.num_samples = len(self.patient_dirs)
            print(f"Loaded {self.num_samples} patients from {data_path}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.use_dummy:
            return self._get_dummy_item(idx)
        
        patient_dir = self.patient_dirs[idx]
        
        try:
            # Load CT volume
            ct_path = patient_dir / "ct_volume.npy"
            ct_volume = np.load(ct_path).astype(np.float32)
            
            # Normalize to [-1, 1]
            ct_volume = self._normalize_volume(ct_volume)
            
            # Resize if needed
            ct_volume = self._resize_volume(ct_volume, self.target_volume_size)
            
            # Load X-ray images (typically 2 views)
            xray_files = sorted(patient_dir.glob("xray_*.npy"))
            xrays = []
            for xray_file in xray_files:
                xray = np.load(xray_file).astype(np.float32)
                xray = self._normalize_image(xray)
                
                # Ensure shape is [C, H, W]
                if xray.ndim == 2:
                    xray = xray[np.newaxis, ...]  # Add channel dimension
                
                xrays.append(xray)
            
            # Stack X-rays: [num_views, C, H, W]
            xrays = np.stack(xrays, axis=0)
            
            # Load metadata if available
            metadata_path = patient_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            return {
                'volume': torch.from_numpy(ct_volume),  # [D, H, W]
                'xrays': torch.from_numpy(xrays),        # [num_views, C, H, W]
                'patient_id': patient_dir.name,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error loading patient {patient_dir.name}: {e}")
            return self._get_dummy_item(idx)
    
    def _get_dummy_item(self, idx):
        """Generate dummy data for testing"""
        D, H, W = self.target_volume_size
        
        # Dummy CT volume with some structure
        volume = np.random.randn(D, H, W).astype(np.float32) * 0.1
        volume = np.clip(volume, -1, 1)
        
        # Dummy X-rays (2 views)
        xrays = np.random.randn(2, 1, 512, 512).astype(np.float32) * 0.1
        xrays = np.clip(xrays, -1, 1)
        
        return {
            'volume': torch.from_numpy(volume),
            'xrays': torch.from_numpy(xrays),
            'patient_id': f'dummy_patient_{idx:03d}',
            'metadata': {}
        }
    
    def _normalize_volume(self, volume):
        """Normalize volume to [-1, 1] range"""
        vmin, vmax = volume.min(), volume.max()
        if vmax > vmin:
            volume = 2 * (volume - vmin) / (vmax - vmin) - 1
        return volume
    
    def _normalize_image(self, image):
        """Normalize image to [-1, 1] range"""
        imin, imax = image.min(), image.max()
        if imax > imin:
            image = 2 * (image - imin) / (imax - imin) - 1
        return image
    
    def _resize_volume(self, volume, target_size):
        """Resize volume to target size using interpolation"""
        # Handle different input shapes
        if volume.ndim == 4:  # [C, D, H, W]
            volume = volume[0]  # Take first channel
        
        D, H, W = volume.shape
        target_D, target_H, target_W = target_size
        
        if (D, H, W) == target_size:
            return volume
        
        # Simple resize using torch interpolate
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        resized = torch.nn.functional.interpolate(
            volume_tensor, 
            size=target_size, 
            mode='trilinear', 
            align_corners=False
        )
        return resized.squeeze(0).squeeze(0).numpy()


def create_dummy_dataset(output_path, num_patients=10, volume_size=(64, 64, 64)):
    """
    Create a dummy dataset for testing purposes.
    
    Args:
        output_path: Path where to save the dummy dataset
        num_patients: Number of dummy patients to create
        volume_size: Size of CT volumes
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy dataset at {output_path}")
    
    for i in range(num_patients):
        patient_dir = output_path / f"patient_{i+1:03d}"
        patient_dir.mkdir(exist_ok=True)
        
        # Create dummy CT volume
        D, H, W = volume_size
        ct_volume = np.random.randn(D, H, W).astype(np.float32) * 100 + 500
        np.save(patient_dir / "ct_volume.npy", ct_volume)
        
        # Create dummy X-ray images (2 views)
        for view_idx in range(2):
            xray = np.random.randn(512, 512).astype(np.float32) * 50 + 200
            np.save(patient_dir / f"xray_{view_idx}.npy", xray)
        
        # Create metadata
        metadata = {
            "patient_id": f"patient_{i+1:03d}",
            "volume_shape": [D, H, W],
            "num_xrays": 2
        }
        with open(patient_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Created {num_patients} dummy patients")


if __name__ == "__main__":
    # Test dataset creation
    test_path = "/tmp/test_drr_dataset"
    create_dummy_dataset(test_path, num_patients=5)
    
    # Test dataset loading
    dataset = PatientDRRDataset(test_path, target_volume_size=(64, 64, 64))
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Volume shape: {sample['volume'].shape}")
    print(f"X-rays shape: {sample['xrays'].shape}")
    print(f"Patient ID: {sample['patient_id']}")
