# 3D Vision and Medical Imaging - Complete Guide

## Table of Contents
- [Introduction](#introduction)
- [Medical Image Formats and Preprocessing](#medical-image-formats-and-preprocessing)
- [3D CNNs](#3d-cnns)
- [3D Medical Image Segmentation](#3d-medical-image-segmentation)
- [Point Cloud Processing](#point-cloud-processing)
- [Data Augmentation for Medical Imaging](#data-augmentation-for-medical-imaging)
- [Transfer Learning for Medical Imaging](#transfer-learning-for-medical-imaging)
- [Object Detection in Medical Images](#object-detection-in-medical-images)
- [Radiomics and Feature Extraction](#radiomics-and-feature-extraction)
- [Evaluation Metrics for Medical Imaging](#evaluation-metrics-for-medical-imaging)
- [Regulatory and Clinical Considerations](#regulatory-and-clinical-considerations)
- [Resources and References](#resources-and-references)

---

## Introduction

### The 3D Vision Landscape

**3D vision** extends traditional 2D computer vision to volumetric data, enabling analysis of spatial relationships across depth. This is critical in:

- **Medical imaging**: CT, MRI, PET scans are inherently 3D
- **Autonomous driving**: LiDAR point clouds for environment mapping
- **Robotics**: 3D object manipulation and scene understanding
- **Video understanding**: Temporal dimension as third axis
- **AR/VR**: Depth estimation and 3D reconstruction

### Medical Imaging Modalities

**Computed Tomography (CT)**
- X-ray based volumetric imaging
- Excellent for bone and lung tissue
- Hounsfield Units (HU) measure radiodensity
- Typical resolution: 0.5-1mm isotropic

**Magnetic Resonance Imaging (MRI)**
- Uses magnetic fields and radio waves
- Superior soft tissue contrast
- Multiple sequences: T1, T2, FLAIR, DWI
- No ionizing radiation

**X-ray**
- 2D projection imaging
- Fast and widely available
- Limited soft tissue contrast

**Ultrasound**
- Real-time imaging using sound waves
- Safe for pregnancy, portable
- Operator-dependent quality

**Positron Emission Tomography (PET)**
- Functional imaging showing metabolism
- Often combined with CT (PET/CT)
- Used for cancer staging

### Why 3D Matters in Medical Imaging

**Volumetric Context**
- Tumors have 3D extent and irregular shapes
- Organ segmentation requires spatial continuity
- Pathology spreads in 3D space

**Spatial Relationships**
- Anatomical structures have consistent 3D topology
- Distance to critical structures affects treatment planning
- Vessel networks require 3D tracking

**Clinical Decision Making**
- Tumor volume is prognostic indicator
- 3D visualization aids surgical planning
- Radiation therapy requires precise 3D targeting

---

## Medical Image Formats and Preprocessing

### DICOM Format

**DICOM (Digital Imaging and Communications in Medicine)** is the standard for medical imaging data.

```python
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_dicom_series(dicom_folder):
    """Load a complete DICOM series from folder."""
    # Get all DICOM files
    dicom_files = sorted(Path(dicom_folder).glob('*.dcm'))

    # Read first file to get metadata
    ref_dicom = pydicom.dcmread(str(dicom_files[0]))

    # Extract image dimensions
    pixel_dims = (int(ref_dicom.Rows), int(ref_dicom.Columns), len(dicom_files))
    pixel_spacing = (float(ref_dicom.PixelSpacing[0]),
                     float(ref_dicom.PixelSpacing[1]),
                     float(ref_dicom.SliceThickness))

    # Create 3D volume
    volume = np.zeros(pixel_dims, dtype=ref_dicom.pixel_array.dtype)

    # Load all slices
    for i, dicom_file in enumerate(dicom_files):
        ds = pydicom.dcmread(str(dicom_file))
        volume[:, :, i] = ds.pixel_array

    return volume, pixel_spacing, ref_dicom

def extract_dicom_metadata(dicom_header):
    """Extract key metadata from DICOM header."""
    metadata = {
        'PatientID': dicom_header.get('PatientID', 'Unknown'),
        'StudyDate': dicom_header.get('StudyDate', 'Unknown'),
        'Modality': dicom_header.get('Modality', 'Unknown'),
        'SeriesDescription': dicom_header.get('SeriesDescription', 'Unknown'),
        'ManufacturerModelName': dicom_header.get('ManufacturerModelName', 'Unknown'),
        'KVP': dicom_header.get('KVP', None),  # For CT
        'SliceThickness': dicom_header.get('SliceThickness', None),
        'PixelSpacing': dicom_header.get('PixelSpacing', None),
    }
    return metadata

# Example usage
volume, spacing, header = load_dicom_series('/path/to/dicom/folder')
metadata = extract_dicom_metadata(header)
print(f"Volume shape: {volume.shape}")
print(f"Spacing (mm): {spacing}")
print(f"Modality: {metadata['Modality']}")
```

### NIfTI Format

**NIfTI (Neuroimaging Informatics Technology Initiative)** is common in research, especially neuroimaging.

```python
import nibabel as nib
import numpy as np

def load_nifti(nifti_path):
    """Load NIfTI file with proper orientation."""
    nii = nib.load(nifti_path)

    # Get image data
    volume = nii.get_fdata()

    # Get affine transformation matrix
    affine = nii.affine

    # Get voxel spacing
    spacing = nii.header.get_zooms()

    return volume, affine, spacing

def save_nifti(volume, affine, output_path):
    """Save volume as NIfTI file."""
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, output_path)

# Example usage
volume, affine, spacing = load_nifti('brain_mri.nii.gz')
print(f"Volume shape: {volume.shape}")
print(f"Voxel spacing: {spacing}")
print(f"Affine matrix:\n{affine}")
```

### CT Windowing

**Windowing** adjusts contrast by mapping HU values to display range.

```python
def apply_ct_window(ct_volume, window_center, window_width):
    """Apply CT windowing (level/width adjustment)."""
    lower_bound = window_center - (window_width / 2)
    upper_bound = window_center + (window_width / 2)

    windowed = np.clip(ct_volume, lower_bound, upper_bound)
    windowed = (windowed - lower_bound) / window_width

    return windowed

def get_preset_window(preset_name):
    """Get common CT window presets."""
    presets = {
        'lung': {'center': -600, 'width': 1500},
        'bone': {'center': 300, 'width': 1500},
        'soft_tissue': {'center': 40, 'width': 400},
        'brain': {'center': 40, 'width': 80},
        'liver': {'center': 30, 'width': 150},
        'mediastinum': {'center': 50, 'width': 350},
    }
    return presets.get(preset_name, {'center': 0, 'width': 400})

# Example: Apply different windows
ct_scan, _, _ = load_dicom_series('/path/to/ct')

lung_window = apply_ct_window(ct_scan, **get_preset_window('lung'))
bone_window = apply_ct_window(ct_scan, **get_preset_window('bone'))
brain_window = apply_ct_window(ct_scan, **get_preset_window('brain'))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(lung_window[:, :, ct_scan.shape[2]//2], cmap='gray')
axes[0].set_title('Lung Window')
axes[1].imshow(bone_window[:, :, ct_scan.shape[2]//2], cmap='gray')
axes[1].set_title('Bone Window')
axes[2].imshow(brain_window[:, :, ct_scan.shape[2]//2], cmap='gray')
axes[2].set_title('Brain Window')
plt.tight_layout()
```

### MRI Normalization Techniques

```python
from scipy.ndimage import zoom
import SimpleITK as sitk

def n4_bias_correction(mri_volume, shrink_factor=4, iterations=50):
    """Apply N4 bias field correction to MRI."""
    # Convert numpy to SimpleITK
    image = sitk.GetImageFromArray(mri_volume)

    # Create mask (non-zero voxels)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)

    # Apply N4 bias correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([iterations] * 4)

    corrected = corrector.Execute(image, mask)

    return sitk.GetArrayFromImage(corrected)

def zscore_normalize(volume, mask=None):
    """Z-score normalization (zero mean, unit variance)."""
    if mask is not None:
        mean = volume[mask > 0].mean()
        std = volume[mask > 0].std()
    else:
        mean = volume.mean()
        std = volume.std()

    normalized = (volume - mean) / (std + 1e-8)
    return normalized

def percentile_normalize(volume, lower=1, upper=99):
    """Normalize based on percentiles to handle outliers."""
    lower_val = np.percentile(volume, lower)
    upper_val = np.percentile(volume, upper)

    volume = np.clip(volume, lower_val, upper_val)
    volume = (volume - lower_val) / (upper_val - lower_val)

    return volume

# Example usage
mri_scan, _, _ = load_nifti('brain_t1.nii.gz')

# Bias correction
corrected_mri = n4_bias_correction(mri_scan)

# Z-score normalization
normalized_mri = zscore_normalize(corrected_mri)

# Percentile normalization (robust to outliers)
percentile_normalized = percentile_normalize(mri_scan)
```

### Resampling to Isotropic Spacing

```python
def resample_to_isotropic(volume, original_spacing, target_spacing=1.0):
    """Resample volume to isotropic spacing."""
    # Calculate zoom factors
    zoom_factors = np.array(original_spacing) / target_spacing

    # Resample
    resampled = zoom(volume, zoom_factors, order=3)  # cubic interpolation

    return resampled

def resample_sitk(volume, original_spacing, target_spacing, interpolation='linear'):
    """Resample using SimpleITK for better quality."""
    # Convert to SimpleITK image
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(original_spacing[::-1])  # ITK uses reverse order

    # Calculate new size
    original_size = np.array(volume.shape[::-1])
    new_size = (original_size * np.array(original_spacing) /
                np.array([target_spacing] * 3)).astype(int)

    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([target_spacing] * 3)
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if interpolation == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolation == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation == 'bspline':
        resampler.SetInterpolator(sitk.sitkBSpline)

    resampled = resampler.Execute(image)
    return sitk.GetArrayFromImage(resampled)

# Example
ct_volume, spacing, _ = load_dicom_series('/path/to/ct')
print(f"Original spacing: {spacing}")
print(f"Original shape: {ct_volume.shape}")

# Resample to 1mm isotropic
isotropic_volume = resample_to_isotropic(ct_volume, spacing, target_spacing=1.0)
print(f"Resampled shape: {isotropic_volume.shape}")
```

### Intensity Normalization and Clipping

```python
def clip_and_normalize_ct(ct_volume, min_hu=-1000, max_hu=400):
    """Clip CT values and normalize to [0, 1]."""
    clipped = np.clip(ct_volume, min_hu, max_hu)
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    return normalized

def standardize_intensity(volume, mean=None, std=None):
    """Standardize to specific mean and std."""
    if mean is None:
        mean = volume.mean()
    if std is None:
        std = volume.std()

    standardized = (volume - volume.mean()) / (volume.std() + 1e-8)
    standardized = standardized * std + mean

    return standardized

# Complete preprocessing pipeline
def preprocess_ct_scan(dicom_folder, target_spacing=1.0, clip_range=(-1000, 400)):
    """Complete CT preprocessing pipeline."""
    # Load DICOM series
    volume, spacing, header = load_dicom_series(dicom_folder)

    # Resample to isotropic
    volume = resample_to_isotropic(volume, spacing, target_spacing)

    # Clip and normalize
    volume = clip_and_normalize_ct(volume, *clip_range)

    return volume
```

---

## 3D CNNs

### 3D Convolution Operation

**3D convolution** extends 2D convolution with an additional spatial dimension:
- **Kernel dimensions**: (D, H, W) instead of (H, W)
- **Parameters**: For C_in input channels, C_out output channels, kernel size k: `C_out * C_in * k * k * k + C_out` (bias)
- **Computational cost**: Significantly higher than 2D (cubic vs quadratic)

### 3D CNN Architectures

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DBlock(nn.Module):
    """3D Convolution block with BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResBlock3D(nn.Module):
    """3D Residual block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    """3D ResNet for volumetric data."""
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock3D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResBlock3D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Example usage
model = ResNet3D(num_classes=3, in_channels=1)
input_volume = torch.randn(2, 1, 64, 64, 64)  # Batch of 2, 64^3 volumes
output = model(input_volume)
print(f"Input shape: {input_volume.shape}")
print(f"Output shape: {output.shape}")
```

### Video Understanding Models

```python
class C3D(nn.Module):
    """C3D network for video classification."""
    def __init__(self, num_classes=101, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits
```

### Memory Management for 3D Volumes

```python
class PatchDataset(torch.utils.data.Dataset):
    """Dataset that extracts patches from 3D volumes."""
    def __init__(self, volume_paths, patch_size=(64, 64, 64),
                 stride=(32, 32, 32), transform=None):
        self.volume_paths = volume_paths
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # Pre-compute patch locations for each volume
        self.patch_locations = []
        for vol_path in volume_paths:
            volume, _, _ = load_nifti(vol_path)
            patches = self._get_patch_locations(volume.shape)
            self.patch_locations.extend([(vol_path, loc) for loc in patches])

    def _get_patch_locations(self, volume_shape):
        """Get all valid patch locations in volume."""
        locations = []
        for z in range(0, volume_shape[0] - self.patch_size[0] + 1, self.stride[0]):
            for y in range(0, volume_shape[1] - self.patch_size[1] + 1, self.stride[1]):
                for x in range(0, volume_shape[2] - self.patch_size[2] + 1, self.stride[2]):
                    locations.append((z, y, x))
        return locations

    def __len__(self):
        return len(self.patch_locations)

    def __getitem__(self, idx):
        vol_path, (z, y, x) = self.patch_locations[idx]

        # Load volume (consider caching for efficiency)
        volume, _, _ = load_nifti(vol_path)

        # Extract patch
        patch = volume[z:z+self.patch_size[0],
                      y:y+self.patch_size[1],
                      x:x+self.patch_size[2]]

        # Add channel dimension
        patch = patch[np.newaxis, ...]

        if self.transform:
            patch = self.transform(patch)

        return torch.FloatTensor(patch)

# Training with patch-based approach
def train_3d_model_with_patches():
    """Train 3D model using patch extraction."""
    # Create dataset
    volume_paths = ['/path/to/volume1.nii.gz', '/path/to/volume2.nii.gz']
    dataset = PatchDataset(volume_paths, patch_size=(64, 64, 64))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                             shuffle=True, num_workers=4)

    # Model
    model = ResNet3D(num_classes=2, in_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_idx, patches in enumerate(dataloader):
            patches = patches.to(device)

            # Forward pass
            outputs = model(patches)

            # For this example, we need labels (not shown)
            # labels = ...
            # loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            # loss.backward()
            optimizer.step()

            # total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

---

## 3D Medical Image Segmentation

### 3D U-Net Architecture

```python
class DoubleConv3D(nn.Module):
    """Double convolution block for 3D U-Net."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation."""
    def __init__(self, in_channels=1, num_classes=2, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder_blocks.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features[:-1]):
            self.decoder_blocks.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv3D(feature*2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)

        # Final convolution
        self.final_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoder_blocks), 2):
            x = self.decoder_blocks[idx](x)  # Upsampling
            skip = skip_connections[idx//2]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[idx+1](x)  # Double conv

        return self.final_conv(x)

# Example usage
model = UNet3D(in_channels=1, num_classes=3)
input_volume = torch.randn(1, 1, 128, 128, 128)
output = model(input_volume)
print(f"Input: {input_volume.shape}, Output: {output.shape}")
```

### V-Net Architecture

```python
class VNetBlock(nn.Module):
    """V-Net residual block."""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.convs = nn.ModuleList()

        for i in range(num_convs):
            if i == 0:
                self.convs.append(nn.Conv3d(in_channels, out_channels,
                                           kernel_size=5, padding=2))
            else:
                self.convs.append(nn.Conv3d(out_channels, out_channels,
                                           kernel_size=5, padding=2))

        self.activation = nn.PReLU(out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        for conv in self.convs:
            x = conv(x)
            x = self.activation(x)

        return x + residual

class VNet(nn.Module):
    """V-Net for 3D medical image segmentation."""
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        # Encoder
        self.enc1 = VNetBlock(in_channels, 16, num_convs=1)
        self.down1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)

        self.enc2 = VNetBlock(32, 32, num_convs=2)
        self.down2 = nn.Conv3d(32, 64, kernel_size=2, stride=2)

        self.enc3 = VNetBlock(64, 64, num_convs=3)
        self.down3 = nn.Conv3d(64, 128, kernel_size=2, stride=2)

        self.enc4 = VNetBlock(128, 128, num_convs=3)
        self.down4 = nn.Conv3d(128, 256, kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = VNetBlock(256, 256, num_convs=3)

        # Decoder
        self.up4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec4 = VNetBlock(256, 128, num_convs=3)

        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = VNetBlock(128, 64, num_convs=3)

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = VNetBlock(64, 32, num_convs=2)

        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = VNetBlock(32, 16, num_convs=1)

        # Output
        self.out = nn.Conv3d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = self.down1(x1)

        x2 = self.enc2(x)
        x = self.down2(x2)

        x3 = self.enc3(x)
        x = self.down3(x3)

        x4 = self.enc4(x)
        x = self.down4(x4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up4(x)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        return self.out(x)
```

### nnU-Net Framework

**nnU-Net** is a self-configuring framework that automatically adapts to any dataset.

```python
# nnU-Net usage example (command-line based)
"""
# 1. Set up environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# 2. Organize data according to nnU-Net format:
# nnUNet_raw/
#   Dataset001_TaskName/
#     imagesTr/  # Training images
#     labelsTr/  # Training labels
#     imagesTs/  # Test images (optional)
#     dataset.json  # Dataset description

# 3. Preprocess the dataset
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# 4. Train the model (3D full resolution)
nnUNetv2_train 1 3d_fullres 0  # Dataset 1, 3D config, fold 0

# 5. Run inference
nnUNetv2_predict -i /path/to/input -o /path/to/output -d 1 -c 3d_fullres
"""

# Creating dataset.json for nnU-Net
import json

def create_nnunet_dataset_json(output_path, task_name, num_training):
    """Create dataset.json for nnU-Net."""
    dataset_json = {
        "name": task_name,
        "description": "Medical image segmentation task",
        "tensorImageSize": "4D",
        "reference": "Your institution",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0",
        "modality": {
            "0": "CT"  # or MRI, etc.
        },
        "labels": {
            "background": 0,
            "organ": 1,
            "tumor": 2
        },
        "numTraining": num_training,
        "numTest": 0,
        "training": [
            {
                "image": f"./imagesTr/case_{i:05d}.nii.gz",
                "label": f"./labelsTr/case_{i:05d}.nii.gz"
            }
            for i in range(num_training)
        ],
        "test": []
    }

    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)

# Example
create_nnunet_dataset_json(
    '/path/to/nnUNet_raw/Dataset001_Liver/dataset.json',
    'LiverSegmentation',
    num_training=100
)
```

### MONAI Framework

```python
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, RandCropByPosNegLabeld
)
from monai.data import Dataset, DataLoader

# Define transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1,
        num_samples=4
    ),
])

# Create dataset
data_dicts = [
    {"image": "img1.nii.gz", "label": "seg1.nii.gz"},
    {"image": "img2.nii.gz", "label": "seg2.nii.gz"},
]

train_ds = Dataset(data=data_dicts, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# Loss and optimizer
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training loop
def train_monai_model(model, train_loader, loss_function, optimizer, num_epochs=10):
    """Train MONAI model."""
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= step
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
```

### Segmentation Loss Functions

```python
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)

        # Flatten
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target = target.contiguous().view(target.size(0), -1)

        # One-hot encode target
        target_one_hot = F.one_hot(target.long(), num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 2, 1).float()

        # Compute Dice
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss

class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss."""
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target.long())
        return self.dice_weight * dice + self.ce_weight * ce

class TverskyLoss(nn.Module):
    """Tversky loss - generalization of Dice, useful for imbalanced data."""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # False positive weight
        self.beta = beta    # False negative weight
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)

        # Flatten
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target = target.contiguous().view(target.size(0), -1)

        # One-hot encode
        target_one_hot = F.one_hot(target.long(), num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(0, 2, 1).float()

        # True positives, false positives, false negatives
        TP = (pred * target_one_hot).sum(dim=2)
        FP = (pred * (1 - target_one_hot)).sum(dim=2)
        FN = ((1 - pred) * target_one_hot).sum(dim=2)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - tversky.mean()

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## Point Cloud Processing

### PointNet Architecture

```python
class TNet(nn.Module):
    """Transformation network for spatial transformation."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity matrix
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x

class PointNet(nn.Module):
    """PointNet for point cloud classification."""
    def __init__(self, num_classes=10, num_points=1024):
        super().__init__()

        # Input transformation
        self.input_transform = TNet(k=3)

        # MLP on points
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # Feature transformation
        self.feature_transform = TNet(k=64)

        # MLP on features
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # Batch norms
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x shape: (batch, 3, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        # Input transformation
        trans_input = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_input)
        x = x.transpose(2, 1)

        # MLP(64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transformation
        trans_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        # MLP(64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling (global feature)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Classification MLP
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x, trans_input, trans_feat

# Example usage
model = PointNet(num_classes=40, num_points=1024)
point_cloud = torch.randn(4, 3, 1024)  # Batch of 4, 1024 points with xyz
output, trans1, trans2 = model(point_cloud)
print(f"Input: {point_cloud.shape}, Output: {output.shape}")
```

### Open3D Library Usage

```python
import open3d as o3d
import numpy as np

def load_and_visualize_point_cloud(file_path):
    """Load and visualize point cloud."""
    # Load point cloud
    pcd = o3d.io.read_point_cloud(file_path)

    print(f"Point cloud has {len(pcd.points)} points")

    # Visualize
    o3d.visualization.draw_geometries([pcd])

    return pcd

def preprocess_point_cloud(pcd, voxel_size=0.05):
    """Preprocess point cloud: downsample, estimate normals."""
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )

    # Remove outliers
    pcd_clean, ind = pcd_down.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    return pcd_clean

def point_cloud_registration(source, target, threshold=0.02):
    """Register two point clouds using ICP."""
    # Initial alignment (identity)
    trans_init = np.eye(4)

    # ICP registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    print(f"Fitness: {reg_p2p.fitness:.4f}")
    print(f"Inlier RMSE: {reg_p2p.inlier_rmse:.4f}")

    return reg_p2p.transformation

def convert_to_pytorch_format(pcd, num_points=1024):
    """Convert Open3D point cloud to PyTorch tensor."""
    points = np.asarray(pcd.points)

    # Sample or pad to fixed number of points
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]

    # Convert to PyTorch
    points_tensor = torch.FloatTensor(points).transpose(0, 1)  # (3, num_points)

    return points_tensor

# Example workflow
# pcd = load_and_visualize_point_cloud('pointcloud.ply')
# pcd_clean = preprocess_point_cloud(pcd)
# points_tensor = convert_to_pytorch_format(pcd_clean)
```

---

## Data Augmentation for Medical Imaging

### MONAI Transforms Pipeline

```python
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, RandRotate90d,
    RandFlipd, RandAffined, RandGaussianNoised,
    RandGaussianSmoothd, RandAdjustContrastd, RandShiftIntensityd,
    RandElasticd, RandCropByPosNegLabeld
)

# Comprehensive augmentation pipeline
train_transforms = Compose([
    # Load data
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),

    # Spatial preprocessing
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),

    # Intensity preprocessing
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000,  # CT min HU
        a_max=400,    # CT max HU
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    # Spatial augmentations
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

    RandAffined(
        keys=["image", "label"],
        prob=0.5,
        rotate_range=(0.1, 0.1, 0.1),  # ~6 degrees
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest"),
        padding_mode="border"
    ),

    RandElasticd(
        keys=["image", "label"],
        prob=0.2,
        sigma_range=(5, 7),
        magnitude_range=(50, 150),
        mode=("bilinear", "nearest")
    ),

    # Intensity augmentations
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
    RandGaussianSmoothd(keys=["image"], prob=0.15, sigma_x=(0.5, 1.0)),
    RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5)),
    RandShiftIntensityd(keys=["image"], prob=0.15, offsets=0.1),

    # Random crop
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1,
        num_samples=4
    ),
])

# Validation transforms (no augmentation)
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
])
```

### TorchIO for Medical Image Augmentation

```python
import torchio as tio

# Define comprehensive augmentation
augmentation = tio.Compose([
    # Spatial transforms
    tio.RandomFlip(axes=('LR',), flip_probability=0.5),
    tio.RandomAffine(
        scales=(0.9, 1.1),
        degrees=10,
        translation=5,
        p=0.5
    ),
    tio.RandomElasticDeformation(
        num_control_points=7,
        max_displacement=7.5,
        locked_borders=2,
        p=0.2
    ),

    # Intensity transforms
    tio.RandomBiasField(coefficients=0.5, p=0.3),
    tio.RandomNoise(std=(0, 0.05), p=0.25),
    tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.25),
    tio.RandomBlur(std=(0, 2), p=0.15),

    # Medical-specific
    tio.RandomSpike(num_spikes=(1, 3), intensity=(0.1, 1), p=0.1),  # MRI artifact
    tio.RandomGhosting(num_ghosts=(4, 10), intensity=(0.5, 1), p=0.1),  # MRI artifact
    tio.RandomMotion(degrees=10, translation=10, num_transforms=2, p=0.1),  # Motion artifact
])

# Create subject
subject = tio.Subject(
    image=tio.ScalarImage('ct_scan.nii.gz'),
    label=tio.LabelMap('segmentation.nii.gz'),
)

# Apply augmentation
augmented = augmentation(subject)

# Create dataset
subjects = [
    tio.Subject(image=tio.ScalarImage(f'image_{i}.nii.gz'),
                label=tio.LabelMap(f'label_{i}.nii.gz'))
    for i in range(100)
]

dataset = tio.SubjectsDataset(subjects, transform=augmentation)

# Patch-based sampling
sampler = tio.data.UniformSampler(patch_size=96)
patches_queue = tio.Queue(
    dataset,
    max_length=40,
    samples_per_volume=10,
    sampler=sampler,
    num_workers=4
)

# DataLoader
from torch.utils.data import DataLoader
patch_loader = DataLoader(patches_queue, batch_size=4)
```

### CutMix/MixUp for 3D

```python
def mixup_3d(x, y, alpha=1.0):
    """MixUp augmentation for 3D volumes."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def cutmix_3d(x, y, alpha=1.0):
    """CutMix augmentation for 3D volumes."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Random bounding box
    D, H, W = x.size()[2:]
    cut_rat = np.sqrt(1. - lam)
    cut_d = int(D * cut_rat)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # Uniform
    cz = np.random.randint(D)
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    bbz1 = np.clip(cz - cut_d // 2, 0, D)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)

    # Apply cutmix
    x[:, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2] = x[index, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2]

    # Adjust lambda
    lam = 1 - ((bbz2 - bbz1) * (bby2 - bby1) * (bbx2 - bbx1) / (D * H * W))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# Training with MixUp/CutMix
def train_with_augmentation(model, dataloader, criterion, optimizer):
    """Train with MixUp or CutMix."""
    model.train()

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Apply MixUp or CutMix randomly
        if np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = mixup_3d(images, labels)
        else:
            images, labels_a, labels_b, lam = cutmix_3d(images, labels)

        # Forward pass
        outputs = model(images)

        # Mixed loss
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Transfer Learning for Medical Imaging

### ImageNet Pretrained Models

Despite domain differences, **ImageNet pretrained models work surprisingly well** for medical imaging:

```python
import torchvision.models as models

def get_pretrained_model_for_medical(num_classes, in_channels=1):
    """Adapt ImageNet pretrained model for medical imaging."""
    # Load pretrained ResNet
    model = models.resnet50(pretrained=True)

    # Modify first conv layer for grayscale or different channels
    if in_channels != 3:
        weight = model.conv1.weight.clone()

        if in_channels == 1:
            # Average RGB weights for grayscale
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1.weight.data = weight.mean(dim=1, keepdim=True)
        else:
            # Replicate or adapt for other channel counts
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1.weight.data = weight.repeat(1, in_channels//3 + 1, 1, 1)[:, :in_channels, :, :]

    # Modify classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

# Example usage
model = get_pretrained_model_for_medical(num_classes=3, in_channels=1)

# Freeze backbone for initial training
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier
for param in model.fc.parameters():
    param.requires_grad = True
```

### Models Genesis - Medical-Specific Pretraining

**Models Genesis** provides self-supervised pretraining on medical images:

```python
# Models Genesis uses self-supervised tasks:
# 1. Local shuffling
# 2. Non-linear transformation
# 3. Inner/outer cutout
# 4. Painting

def models_genesis_pretraining():
    """Pseudo-code for Models Genesis approach."""
    # Download pretrained weights
    # url = 'https://zenodo.org/record/3431873/files/Genesis_Chest_CT.pt'

    # Load 3D UNet
    model = UNet3D(in_channels=1, num_classes=1)

    # Load pretrained weights
    # model.load_state_dict(torch.load('Genesis_Chest_CT.pt'))

    # Fine-tune on your task
    # Replace final layer
    # model.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    return model

# Transfer learning strategy
def finetune_with_genesis(model, train_loader, num_epochs=50):
    """Fine-tune Models Genesis pretrained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Stage 1: Train only classifier (first 10 epochs)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.final_conv.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = DiceCELoss()

    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Stage 2: Unfreeze all and train with lower learning rate
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10, num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Self-Supervised Pretraining

```python
class ContrastiveLearning3D(nn.Module):
    """Contrastive learning for 3D medical images."""
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """NT-Xent loss for contrastive learning."""
    batch_size = z_i.size(0)

    # Concatenate representations
    z = torch.cat([z_i, z_j], dim=0)

    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature

    # Create labels
    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -float('inf'))

    # Compute loss
    loss = F.cross_entropy(sim, labels)

    return loss

# Self-supervised pretraining loop
def pretrain_contrastive(model, dataloader, num_epochs=100):
    """Pretrain model with contrastive learning."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images in dataloader:
            # Apply two different augmentations
            images_i = augmentation1(images).to(device)
            images_j = augmentation2(images).to(device)

            # Get representations
            z_i = model(images_i)
            z_j = model(images_j)

            # Compute loss
            loss = nt_xent_loss(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

### Few-Shot Learning

```python
class PrototypicalNetwork(nn.Module):
    """Prototypical networks for few-shot learning."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support_images, support_labels, query_images, n_way, k_shot):
        """
        support_images: (n_way * k_shot, C, D, H, W)
        support_labels: (n_way * k_shot,)
        query_images: (n_query, C, D, H, W)
        """
        # Encode support and query
        support_features = self.encoder(support_images)
        query_features = self.encoder(query_images)

        # Compute prototypes (class centroids)
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        # Compute distances to prototypes
        distances = torch.cdist(query_features, prototypes)

        # Convert to probabilities (negative distance)
        logits = -distances

        return logits

# Few-shot training
def train_few_shot(model, dataloader, n_way=5, k_shot=5, n_query=15):
    """Train with few-shot episodes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()

        for batch in dataloader:
            # Sample episode
            support_images, support_labels, query_images, query_labels = sample_episode(
                batch, n_way, k_shot, n_query
            )

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            # Forward pass
            logits = model(support_images, support_labels, query_images, n_way, k_shot)

            # Compute loss
            loss = criterion(logits, query_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Object Detection in Medical Images

### 3D Anchor-Based Detection

```python
class AnchorGenerator3D:
    """Generate 3D anchors for object detection."""
    def __init__(self, sizes=[32, 64, 128], aspect_ratios=[0.5, 1.0, 2.0]):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def generate_anchors(self, feature_map_size, stride):
        """Generate anchors for a feature map."""
        fz, fy, fx = feature_map_size
        anchors = []

        for z in range(fz):
            for y in range(fy):
                for x in range(fx):
                    cz = (z + 0.5) * stride
                    cy = (y + 0.5) * stride
                    cx = (x + 0.5) * stride

                    for size in self.sizes:
                        for ratio in self.aspect_ratios:
                            # Compute anchor dimensions
                            w = size
                            h = size * ratio
                            d = size / ratio

                            # [cz, cy, cx, d, h, w]
                            anchors.append([cz, cy, cx, d, h, w])

        return torch.FloatTensor(anchors)

class RetinaNet3D(nn.Module):
    """3D RetinaNet for medical object detection."""
    def __init__(self, num_classes=2, num_anchors=9):
        super().__init__()

        # Backbone
        self.backbone = ResNet3D(num_classes=1, in_channels=1)

        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # FPN
        self.fpn = FeaturePyramidNetwork3D([128, 256, 512], 256)

        # Classification subnet
        self.cls_subnet = nn.Sequential(
            *[Conv3DBlock(256, 256) for _ in range(4)],
            nn.Conv3d(256, num_anchors * num_classes, kernel_size=3, padding=1)
        )

        # Regression subnet
        self.reg_subnet = nn.Sequential(
            *[Conv3DBlock(256, 256) for _ in range(4)],
            nn.Conv3d(256, num_anchors * 6, kernel_size=3, padding=1)  # 6 for 3D bbox
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # FPN
        fpn_features = self.fpn(features)

        # Predictions
        cls_outputs = []
        reg_outputs = []

        for feat in fpn_features:
            cls_outputs.append(self.cls_subnet(feat))
            reg_outputs.append(self.reg_subnet(feat))

        return cls_outputs, reg_outputs

class FocalLoss3D(nn.Module):
    """Focal loss for 3D object detection."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred: (N, C)
        # target: (N,) class indices

        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        return focal_loss.mean()
```

### nnDetection Framework

```python
# nnDetection usage (similar to nnU-Net)
"""
# 1. Install nnDetection
pip install git+https://github.com/MIC-DKFZ/nnDetection.git

# 2. Set environment variables
export det_data="/path/to/nnDetection_raw"
export det_models="/path/to/nnDetection_models"

# 3. Prepare data in COCO format
# Structure:
# nnDetection_raw/
#   Task000_LIDC/
#     imagesTr/
#     labelsTr/  # JSON files with bounding boxes
#     dataset.json

# 4. Preprocess
nndet_prep 000 -p 8

# 5. Train
nndet_train 000 RetinaUNetV0_D3V001 0

# 6. Evaluate
nndet_eval 000 RetinaUNetV0_D3V001 --fold 0
"""

# Creating dataset for nnDetection
import json

def create_nndetection_annotation(image_id, boxes, labels, image_shape):
    """Create annotation in nnDetection format."""
    annotation = {
        "image_id": image_id,
        "boxes": boxes.tolist(),  # List of [z1, y1, x1, z2, y2, x2]
        "labels": labels.tolist(),  # List of class indices
        "size": image_shape  # [D, H, W]
    }

    return annotation

# Save as JSON
annotations = []
for i, (boxes, labels) in enumerate(detection_data):
    ann = create_nndetection_annotation(
        image_id=f"case_{i:05d}",
        boxes=boxes,
        labels=labels,
        image_shape=(512, 512, 512)
    )
    annotations.append(ann)

# with open('annotations_train.json', 'w') as f:
#     json.dump(annotations, f)
```

### FROC Metric

```python
def compute_froc(predictions, ground_truth, iou_threshold=0.1, num_points=9):
    """
    Compute Free-Response ROC (FROC) for lesion detection.

    FROC plots sensitivity vs. false positives per image.
    """
    # Sort predictions by confidence
    pred_boxes = []
    pred_scores = []
    pred_image_ids = []

    for img_id, preds in predictions.items():
        for box, score in zip(preds['boxes'], preds['scores']):
            pred_boxes.append(box)
            pred_scores.append(score)
            pred_image_ids.append(img_id)

    # Sort by score
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    pred_image_ids = [pred_image_ids[i] for i in sorted_indices]

    # Compute metrics at different thresholds
    sensitivities = []
    fps_per_image = []

    num_images = len(ground_truth)
    num_lesions = sum(len(gt['boxes']) for gt in ground_truth.values())

    for threshold_idx in range(num_points):
        # Keep top predictions
        n_preds = int(len(pred_boxes) * (threshold_idx + 1) / num_points)

        tp = 0
        fp = 0
        matched_gts = set()

        for i in range(n_preds):
            box = pred_boxes[i]
            img_id = pred_image_ids[i]

            # Check IoU with ground truth
            gt_boxes = ground_truth[img_id]['boxes']
            max_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou_3d(box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = gt_idx

            # Check if match
            if max_iou >= iou_threshold:
                gt_key = (img_id, best_gt_idx)
                if gt_key not in matched_gts:
                    tp += 1
                    matched_gts.add(gt_key)
                else:
                    fp += 1
            else:
                fp += 1

        sensitivity = tp / num_lesions if num_lesions > 0 else 0
        fp_per_img = fp / num_images

        sensitivities.append(sensitivity)
        fps_per_image.append(fp_per_img)

    return sensitivities, fps_per_image

def compute_iou_3d(box1, box2):
    """Compute 3D IoU between two boxes."""
    # box format: [z1, y1, x1, z2, y2, x2]
    z1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x1_max = max(box1[2], box2[2])

    z2_min = min(box1[3], box2[3])
    y2_min = min(box1[4], box2[4])
    x2_min = min(box1[5], box2[5])

    # Intersection
    inter_d = max(0, z2_min - z1_max)
    inter_h = max(0, y2_min - y1_max)
    inter_w = max(0, x2_min - x1_max)
    intersection = inter_d * inter_h * inter_w

    # Union
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = vol1 + vol2 - intersection

    iou = intersection / union if union > 0 else 0
    return iou
```

---

## Radiomics and Feature Extraction

### Radiomic Features

**Radiomics** extracts quantitative features from medical images for prediction tasks.

**Feature categories:**
- **Shape features**: Volume, surface area, sphericity, compactness
- **First-order features**: Mean, median, variance, skewness, kurtosis, entropy
- **Texture features**:
  - GLCM (Gray Level Co-occurrence Matrix)
  - GLRLM (Gray Level Run Length Matrix)
  - GLSZM (Gray Level Size Zone Matrix)
  - NGTDM (Neighborhood Gray Tone Difference Matrix)

### PyRadiomics Library

```python
from radiomics import featureextractor
import SimpleITK as sitk

def extract_radiomic_features(image_path, mask_path):
    """Extract radiomic features using PyRadiomics."""
    # Initialize extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Configure
    extractor.enableAllFeatures()

    # Optional: specific settings
    settings = {
        'binWidth': 25,
        'resampledPixelSpacing': [1, 1, 1],  # Isotropic resampling
        'interpolator': sitk.sitkBSpline,
        'normalize': True,
        'normalizeScale': 100
    }
    extractor.settings.update(settings)

    # Extract features
    result = extractor.execute(image_path, mask_path)

    # Parse results
    features = {}
    for key, value in result.items():
        if not key.startswith('diagnostics'):
            features[key] = value

    return features

# Example usage
# features = extract_radiomic_features('ct_image.nii.gz', 'tumor_mask.nii.gz')
# print(f"Extracted {len(features)} features")
# print("Sample features:", list(features.keys())[:5])

def extract_features_batch(image_mask_pairs):
    """Extract features for multiple cases."""
    all_features = []

    for image_path, mask_path in image_mask_pairs:
        try:
            features = extract_radiomic_features(image_path, mask_path)
            features['case_id'] = image_path  # Track case
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(all_features)

    return df

# Specific feature extraction
def extract_shape_features(image_path, mask_path):
    """Extract only shape features."""
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Enable only shape features
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape')

    result = extractor.execute(image_path, mask_path)

    shape_features = {k: v for k, v in result.items()
                     if k.startswith('original_shape')}

    return shape_features

def extract_texture_features(image_path, mask_path):
    """Extract texture features (GLCM, GLRLM, etc.)."""
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Enable texture features
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')

    result = extractor.execute(image_path, mask_path)

    texture_features = {k: v for k, v in result.items()
                       if not k.startswith('diagnostics')}

    return texture_features
```

### Feature Selection for Radiomics

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def select_robust_features(features_df, labels, k=20):
    """Select most informative radiomic features."""
    # Remove non-feature columns
    X = features_df.drop(['case_id'], axis=1, errors='ignore')

    # Handle missing values
    X = X.fillna(X.mean())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection with ANOVA F-test
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, labels)

    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()

    print(f"Selected {len(selected_features)} features:")
    for feat in selected_features:
        print(f"  - {feat}")

    return selected_features, X_selected

def remove_correlated_features(features_df, threshold=0.95):
    """Remove highly correlated features."""
    X = features_df.select_dtypes(include=[np.number])

    # Compute correlation matrix
    corr_matrix = X.corr().abs()

    # Select upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [column for column in upper.columns
               if any(upper[column] > threshold)]

    print(f"Dropping {len(to_drop)} highly correlated features")

    return features_df.drop(to_drop, axis=1)

# Dimensionality reduction
def apply_pca_to_radiomics(features_df, n_components=10):
    """Apply PCA to radiomic features."""
    X = features_df.select_dtypes(include=[np.number])
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    return X_pca, pca
```

### Combining Radiomics with Deep Learning

```python
class RadiomicsCNNFusion(nn.Module):
    """Fusion model combining radiomics and deep features."""
    def __init__(self, cnn_encoder, num_radiomic_features, num_classes):
        super().__init__()

        self.cnn_encoder = cnn_encoder

        # Radiomics branch
        self.radiomic_branch = nn.Sequential(
            nn.Linear(num_radiomic_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(512 + 64, 256),  # CNN features + radiomic features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, radiomic_features):
        # CNN features
        cnn_features = self.cnn_encoder(image)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)

        # Radiomic features
        radiomic_embed = self.radiomic_branch(radiomic_features)

        # Concatenate
        combined = torch.cat([cnn_features, radiomic_embed], dim=1)

        # Final prediction
        output = self.fusion(combined)

        return output

# Training with fusion model
def train_fusion_model(model, train_loader, num_epochs=50):
    """Train radiomics-CNN fusion model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, radiomic_feats, labels in train_loader:
            images = images.to(device)
            radiomic_feats = radiomic_feats.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, radiomic_feats)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

---

## Evaluation Metrics for Medical Imaging

### Dice Similarity Coefficient

```python
def dice_coefficient(pred, target, smooth=1.0):
    """
    Compute Dice Similarity Coefficient (DSC).

    DSC = 2 * |X intersection Y| / (|X| + |Y|)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice

def dice_per_class(pred, target, num_classes):
    """Compute Dice for each class."""
    dice_scores = []

    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).astype(float)
        target_class = (target == class_idx).astype(float)

        dice = dice_coefficient(pred_class, target_class)
        dice_scores.append(dice)

    return dice_scores
```

### Hausdorff Distance

```python
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

def hausdorff_distance(pred, target):
    """
    Compute Hausdorff Distance between two segmentations.

    HD = max(h(pred, target), h(target, pred))
    where h(A, B) = max_{a in A} min_{b in B} ||a - b||
    """
    # Get surface points
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')

    # Compute directed Hausdorff distances
    hd1 = directed_hausdorff(pred_points, target_points)[0]
    hd2 = directed_hausdorff(target_points, pred_points)[0]

    return max(hd1, hd2)

def hausdorff_distance_95(pred, target):
    """
    Compute 95th percentile Hausdorff Distance (more robust).
    """
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')

    # Compute distances from pred to target
    distances_pred_to_target = []
    for point in pred_points:
        min_dist = np.min(np.linalg.norm(target_points - point, axis=1))
        distances_pred_to_target.append(min_dist)

    # Compute distances from target to pred
    distances_target_to_pred = []
    for point in target_points:
        min_dist = np.min(np.linalg.norm(pred_points - point, axis=1))
        distances_target_to_pred.append(min_dist)

    # 95th percentile
    hd95_1 = np.percentile(distances_pred_to_target, 95)
    hd95_2 = np.percentile(distances_target_to_pred, 95)

    return max(hd95_1, hd95_2)
```

### Average Surface Distance

```python
def average_surface_distance(pred, target, spacing=(1.0, 1.0, 1.0)):
    """
    Compute Average Surface Distance (ASD).

    Average distance between surfaces of segmentations.
    """
    # Get surfaces (boundary voxels)
    from scipy.ndimage import binary_erosion

    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)

    # Get surface points
    pred_points = np.argwhere(pred_border) * spacing
    target_points = np.argwhere(target_border) * spacing

    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')

    # Distance transform
    target_dt = distance_transform_edt(~target_border) * spacing[0]
    pred_dt = distance_transform_edt(~pred_border) * spacing[0]

    # Average distances
    distances_pred_to_target = target_dt[pred_border]
    distances_target_to_pred = pred_dt[target_border]

    asd = (distances_pred_to_target.sum() + distances_target_to_pred.sum()) / \
          (len(distances_pred_to_target) + len(distances_target_to_pred))

    return asd
```

### Comprehensive Evaluation

```python
def evaluate_segmentation(pred, target, spacing=(1.0, 1.0, 1.0)):
    """Comprehensive segmentation evaluation."""
    metrics = {}

    # Dice coefficient
    metrics['dice'] = dice_coefficient(pred, target)

    # Hausdorff distances
    metrics['hausdorff'] = hausdorff_distance(pred, target)
    metrics['hausdorff_95'] = hausdorff_distance_95(pred, target)

    # Average surface distance
    metrics['asd'] = average_surface_distance(pred, target, spacing)

    # Sensitivity and Specificity
    tp = np.sum((pred == 1) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))

    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Volume similarity
    vol_pred = np.sum(pred)
    vol_target = np.sum(target)
    metrics['volume_similarity'] = 1 - abs(vol_pred - vol_target) / (vol_pred + vol_target)

    return metrics

# Statistical testing
def compare_models_statistically(results_model_a, results_model_b, metric='dice'):
    """Compare two models using paired t-test."""
    from scipy import stats

    scores_a = [r[metric] for r in results_model_a]
    scores_b = [r[metric] for r in results_model_b]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

    print(f"Model A mean {metric}: {np.mean(scores_a):.4f} +/- {np.std(scores_a):.4f}")
    print(f"Model B mean {metric}: {np.mean(scores_b):.4f} +/- {np.std(scores_b):.4f}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        if np.mean(scores_a) > np.mean(scores_b):
            print("Model A is significantly better")
        else:
            print("Model B is significantly better")
    else:
        print("No significant difference")

    return t_stat, p_value
```

### Cohen's Kappa

```python
def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa for inter-rater agreement.

    Kappa = (Po - Pe) / (1 - Pe)
    where Po is observed agreement and Pe is expected agreement by chance.
    """
    rater1_flat = rater1.flatten()
    rater2_flat = rater2.flatten()

    # Observed agreement
    po = np.mean(rater1_flat == rater2_flat)

    # Expected agreement
    classes = np.unique(np.concatenate([rater1_flat, rater2_flat]))
    pe = 0
    for cls in classes:
        p1 = np.mean(rater1_flat == cls)
        p2 = np.mean(rater2_flat == cls)
        pe += p1 * p2

    # Kappa
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0

    return kappa
```

---

## Regulatory and Clinical Considerations

### FDA Approval Pathway

**Three main pathways for medical device approval:**

**510(k) Premarket Notification**
- Device is "substantially equivalent" to existing device
- Most common pathway (~90% of devices)
- Requires: predicate device, performance testing, clinical data (sometimes)
- Review time: ~90 days

**De Novo Classification**
- Novel device with low/moderate risk
- No predicate device exists
- More stringent than 510(k)
- Creates new device category

**Premarket Approval (PMA)**
- High-risk devices (Class III)
- Most rigorous pathway
- Requires: extensive clinical trials, manufacturing data, risk analysis
- Review time: ~180 days to years

### Required Documentation

```python
# Example validation report structure
validation_report = {
    "Executive Summary": {
        "Device name": "AI-based CT Lesion Detection",
        "Intended use": "Aid radiologists in detecting lung nodules",
        "Classification": "Class II, 510(k) exempt"
    },

    "Algorithm Description": {
        "Architecture": "3D ResNet with FPN",
        "Training data": "10,000 CT scans from 5 institutions",
        "Preprocessing": "Resampling to 1mm, HU clipping [-1000, 400]",
        "Performance metrics": "Sensitivity, specificity, FROC"
    },

    "Validation Results": {
        "Internal validation": {
            "Dataset": "2000 cases",
            "Sensitivity": "95% (95% CI: 93-97%)",
            "Specificity": "90% (95% CI: 88-92%)",
            "AUC": "0.96"
        },
        "External validation": {
            "Dataset": "500 cases from different institution",
            "Sensitivity": "92% (95% CI: 89-95%)",
            "Specificity": "88% (95% CI: 85-91%)"
        }
    },

    "Clinical Validation": {
        "Study design": "Reader study with 10 radiologists",
        "Standalone performance": "Sensitivity 92%, FP rate 0.5 per case",
        "Reader without AI": "Sensitivity 85%, FP rate 0.8 per case",
        "Reader with AI": "Sensitivity 94%, FP rate 0.4 per case"
    },

    "Bias Analysis": {
        "Age groups": "No significant difference (p>0.05)",
        "Gender": "No significant difference (p>0.05)",
        "Ethnicity": "Evaluated across 4 groups, consistent performance",
        "Scanner manufacturer": "Tested on GE, Siemens, Philips"
    },

    "Risk Analysis": {
        "False negatives": "Missed cancer - HIGH RISK - Mitigated by dual reading",
        "False positives": "Unnecessary follow-up - MODERATE RISK",
        "Software bugs": "MODERATE RISK - Extensive testing, version control"
    }
}
```

### CE Marking (EU Medical Device Regulation)

**Requirements under MDR (Medical Device Regulation):**
- Clinical evaluation report
- Risk management (ISO 14971)
- Quality management system (ISO 13485)
- Technical documentation
- Post-market surveillance plan

### Bias and Fairness

```python
def evaluate_fairness(predictions, labels, protected_attributes):
    """
    Evaluate model fairness across demographic groups.

    protected_attributes: dict with keys like 'age', 'gender', 'ethnicity'
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    results = {}

    for attr_name, attr_values in protected_attributes.items():
        group_aucs = {}
        group_accs = {}

        unique_groups = np.unique(attr_values)

        for group in unique_groups:
            mask = attr_values == group
            group_preds = predictions[mask]
            group_labels = labels[mask]

            if len(np.unique(group_labels)) > 1:  # Check both classes present
                auc = roc_auc_score(group_labels, group_preds)
                acc = accuracy_score(group_labels, group_preds > 0.5)

                group_aucs[group] = auc
                group_accs[group] = acc

        # Compute disparity
        auc_values = list(group_aucs.values())
        disparity = max(auc_values) - min(auc_values) if len(auc_values) > 1 else 0

        results[attr_name] = {
            'group_aucs': group_aucs,
            'group_accuracies': group_accs,
            'disparity': disparity
        }

        print(f"\n{attr_name.upper()} Analysis:")
        for group, auc in group_aucs.items():
            print(f"  {group}: AUC = {auc:.3f}, Acc = {group_accs[group]:.3f}")
        print(f"  AUC Disparity: {disparity:.3f}")

    return results
```

### Explainability

```python
class GradCAM3D:
    """Grad-CAM for 3D medical images."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

# Usage
# model = ResNet3D(num_classes=2)
# target_layer = model.layer4[-1]  # Last conv layer
# grad_cam = GradCAM3D(model, target_layer)
#
# input_volume = torch.randn(1, 1, 64, 64, 64)
# cam = grad_cam(input_volume)
```

### Data Privacy and De-identification

```python
def deidentify_dicom(dicom_file, output_file):
    """Remove PHI (Protected Health Information) from DICOM."""
    ds = pydicom.dcmread(dicom_file)

    # Tags to remove/anonymize
    tags_to_remove = [
        'PatientName',
        'PatientID',
        'PatientBirthDate',
        'PatientSex',
        'PatientAge',
        'InstitutionName',
        'InstitutionAddress',
        'ReferringPhysicianName',
        'PerformingPhysicianName',
        'StudyDate',
        'StudyTime',
        'SeriesDate',
        'SeriesTime',
        'AccessionNumber'
    ]

    for tag in tags_to_remove:
        if tag in ds:
            if tag == 'PatientID':
                # Replace with anonymous ID
                ds.PatientID = f"ANON_{hash(ds.PatientID) % 1000000:06d}"
            else:
                delattr(ds, tag)

    # Save
    ds.save_as(output_file)

# HIPAA Safe Harbor method
def check_hipaa_compliance(metadata):
    """Check if metadata meets HIPAA Safe Harbor requirements."""
    phi_identifiers = [
        'names', 'geographic_subdivisions', 'dates', 'phone_numbers',
        'fax_numbers', 'email_addresses', 'ssn', 'medical_record_numbers',
        'health_plan_numbers', 'account_numbers', 'license_numbers',
        'vehicle_identifiers', 'device_identifiers', 'urls', 'ip_addresses',
        'biometric_identifiers', 'photos', 'other_unique_identifiers'
    ]

    violations = []
    for identifier in phi_identifiers:
        if identifier in metadata and metadata[identifier]:
            violations.append(identifier)

    if violations:
        print(f"HIPAA violations found: {violations}")
        return False
    else:
        print("HIPAA Safe Harbor compliant")
        return True
```

---

## Resources and References

### Key Libraries

**Medical Imaging Processing:**
- **MONAI** (Medical Open Network for AI): https://monai.io/
  - Purpose-built for medical imaging deep learning
  - Includes datasets, transforms, networks, metrics
  - Install: `pip install monai`

- **TorchIO**: https://torchio.readthedocs.io/
  - Medical image preprocessing and augmentation
  - Built on PyTorch
  - Install: `pip install torchio`

- **SimpleITK**: https://simpleitk.org/
  - Medical image registration, segmentation
  - Wraps ITK with simple Python interface
  - Install: `pip install SimpleITK`

- **PyRadiomics**: https://pyradiomics.readthedocs.io/
  - Extract radiomic features
  - IBSI compliant
  - Install: `pip install pyradiomics`

**DICOM/NIfTI Handling:**
- **pydicom**: Read/write DICOM files
- **nibabel**: Read/write NIfTI, MINC, MGH formats
- **dcm2niix**: Convert DICOM to NIfTI (command-line)

**Frameworks:**
- **nnU-Net**: https://github.com/MIC-DKFZ/nnUNet
  - Self-configuring segmentation
  - State-of-the-art on many benchmarks

- **nnDetection**: https://github.com/MIC-DKFZ/nnDetection
  - Self-configuring 3D object detection

**Point Cloud:**
- **Open3D**: http://www.open3d.org/
  - Point cloud processing, visualization
  - Install: `pip install open3d`

### Public Datasets

**General Medical Imaging:**
- **Medical Segmentation Decathlon**: 10 segmentation tasks
  - http://medicaldecathlon.com/
  - Liver, brain, heart, lung, pancreas, etc.

- **Grand Challenge**: https://grand-challenge.org/
  - 100+ medical imaging challenges
  - Benchmarks for various tasks

**Chest Imaging:**
- **NIH ChestX-ray14**: 112,120 chest X-rays with 14 diseases
  - https://nihcc.app.box.com/v/ChestXray-NIHCC

- **MIMIC-CXR**: 377,110 chest X-rays with reports
  - https://physionet.org/content/mimic-cxr/

- **RSNA Pneumonia Detection**: Chest X-ray pneumonia
  - https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

- **COVID-19 Image Data Collection**: CT and X-ray
  - https://github.com/ieee8023/covid-chestxray-dataset

**Brain Imaging:**
- **BraTS** (Brain Tumor Segmentation): Annual challenge
  - http://braintumorsegmentation.org/
  - MRI with glioma segmentations

- **OASIS** (Open Access Series of Imaging Studies): Brain MRI
  - https://www.oasis-brains.org/

- **Human Connectome Project**: High-resolution brain MRI
  - https://www.humanconnectome.org/

**Lung Imaging:**
- **LUNA16** (Lung Nodule Analysis): CT with nodule annotations
  - https://luna16.grand-challenge.org/

- **LIDC-IDRI**: 1018 lung CT scans with annotations
  - https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

**Liver Imaging:**
- **LiTS** (Liver Tumor Segmentation): CT with liver/tumor masks
  - https://competitions.codalab.org/competitions/17094

**Multi-Organ:**
- **TCIA** (The Cancer Imaging Archive): Diverse cancer imaging
  - https://www.cancerimagingarchive.net/

- **UK Biobank**: 100,000+ participants, multiple modalities
  - https://www.ukbiobank.ac.uk/

### Key Competitions

**Annual Challenges:**
- **MICCAI**: Medical Image Computing and Computer Assisted Intervention
  - Held every September
  - Multiple challenges on segmentation, detection, registration

- **RSNA**: Radiological Society of North America
  - Annual Kaggle competitions
  - Focus on clinically relevant tasks

- **ISBI**: International Symposium on Biomedical Imaging
  - Various challenges

**Competition Platforms:**
- **grand-challenge.org**: Medical imaging challenges
- **Kaggle**: RSNA and other medical competitions
- **CodaLab**: Academic competition hosting

### Key Papers by Topic

**3D CNNs:**
- Cicek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" (2016)
- Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (2016)
- Tran et al., "Learning Spatiotemporal Features with 3D Convolutional Networks" (C3D, 2015)

**Medical Image Segmentation:**
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation" (2021)
- Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018)

**Point Clouds:**
- Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" (2017)
- Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (2017)

**Transfer Learning:**
- Zhou et al., "Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis" (2019)
- Raghu et al., "Transfusion: Understanding Transfer Learning for Medical Imaging" (2019)

**Object Detection:**
- Baumgartner et al., "nnDetection: A Self-configuring Method for Medical Object Detection" (2021)
- Ding et al., "DeepLesion: automated mining of large-scale lesion annotations" (2018)

**Radiomics:**
- Gillies et al., "Radiomics: Images Are More than Pictures, They Are Data" (2016)
- Lambin et al., "Radiomics: the bridge between medical imaging and personalized medicine" (2017)

**Regulatory/Clinical:**
- Topol, "High-performance medicine: the convergence of human and artificial intelligence" (2019)
- Char et al., "Implementing Machine Learning in Health Care - Addressing Ethical Challenges" (2018)
- FDA, "Clinical Performance Assessment: Considerations for Computer-Assisted Detection Devices" (2020)

### Online Courses and Tutorials

**Courses:**
- **Coursera - AI for Medical Diagnosis** (deeplearning.ai)
- **Fast.ai Medical Imaging** (free online course)
- **MONAI Bootcamp** (YouTube playlist)

**Tutorials:**
- MONAI tutorials: https://github.com/Project-MONAI/tutorials
- TorchIO examples: https://torchio.readthedocs.io/examples/
- nnU-Net documentation: https://github.com/MIC-DKFZ/nnUNet

**Books:**
- "Deep Learning for Medical Image Analysis" (Zhou et al., 2017)
- "Medical Image Analysis" (Deserno, 2011)
- "Handbook of Medical Image Computing and Computer Assisted Intervention" (Zhou et al., 2020)

---

**End of 3D Vision and Medical Imaging Guide**

This comprehensive guide covers the essential aspects of 3D vision and medical imaging, from data preprocessing and model architectures to evaluation metrics and regulatory considerations. The field continues to evolve rapidly, with new methods and datasets emerging regularly. Always validate thoroughly and consider clinical relevance when developing medical AI systems.

