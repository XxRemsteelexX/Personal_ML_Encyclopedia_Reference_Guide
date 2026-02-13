# 3D Vision and Medical Imaging

## Table of Contents

- [Introduction](#introduction)
- [3D Data Formats](#3d-data-formats)
  - [Volumetric Grids (Voxels)](#volumetric-grids-voxels)
  - [Point Clouds](#point-clouds)
  - [Meshes](#meshes)
  - [DICOM Format](#dicom-format)
  - [NIfTI Format](#nifti-format)
- [Medical Image Preprocessing](#medical-image-preprocessing)
  - [Hounsfield Unit Windowing](#hounsfield-unit-windowing)
  - [Spacing Normalization](#spacing-normalization)
  - [Intensity Normalization](#intensity-normalization)
  - [Brain Extraction and Skull Stripping](#brain-extraction-and-skull-stripping)
  - [N4 Bias Field Correction](#n4-bias-field-correction)
  - [Registration and Atlas Alignment](#registration-and-atlas-alignment)
- [3D CNN Architectures](#3d-cnn-architectures)
  - [3D Convolutions](#3d-convolutions)
  - [3D ResNet](#3d-resnet)
  - [3D DenseNet](#3d-densenet)
  - [SE-ResNet3D](#se-resnet3d)
  - [I3D Inflated 3D ConvNets](#i3d-inflated-3d-convnets)
  - [C3D for Video Understanding](#c3d-for-video-understanding)
  - [Memory Considerations](#memory-considerations)
- [3D U-Net for Medical Segmentation](#3d-u-net-for-medical-segmentation)
  - [Standard 3D U-Net](#standard-3d-u-net)
  - [V-Net](#v-net)
  - [Attention U-Net](#attention-u-net)
  - [nnU-Net](#nnu-net)
  - [MONAI Framework](#monai-framework)
- [Point Cloud Networks](#point-cloud-networks)
  - [PointNet](#pointnet)
  - [PointNet++](#pointnet-1)
  - [DGCNN](#dgcnn)
  - [Point Transformer](#point-transformer)
  - [PointPillars](#pointpillars)
- [3D Object Detection](#3d-object-detection)
  - [VoxelNet](#voxelnet)
  - [PointPillars for Detection](#pointpillars-for-detection)
  - [CenterPoint](#centerpoint)
  - [SECOND](#second)
  - [BEV Methods](#bev-methods)
- [Medical Image Classification](#medical-image-classification)
  - [Patch-Based Classification](#patch-based-classification)
  - [Multi-Instance Learning](#multi-instance-learning)
  - [Transfer Learning 2D to 3D](#transfer-learning-2d-to-3d)
  - [Class Imbalance Handling](#class-imbalance-handling)
- [Radiomics and Feature Extraction](#radiomics-and-feature-extraction)
  - [PyRadiomics](#pyradiomics)
  - [Texture Features](#texture-features)
  - [Shape and First-Order Features](#shape-and-first-order-features)
- [Neural Radiance Fields](#neural-radiance-fields)
  - [NeRF Architecture](#nerf-architecture)
  - [Positional Encoding](#positional-encoding)
  - [Instant-NGP](#instant-ngp)
  - [3D Gaussian Splatting](#3d-gaussian-splatting)
- [Data Augmentation for 3D and Medical Imaging](#data-augmentation-for-3d-and-medical-imaging)
- [Regulatory and Clinical Considerations](#regulatory-and-clinical-considerations)
  - [FDA Clearance Pathways](#fda-clearance-pathways)
  - [EU MDR and AI Act](#eu-mdr-and-ai-act)
  - [Explainability for Clinical AI](#explainability-for-clinical-ai)
  - [Dataset Bias and Fairness](#dataset-bias-and-fairness)
- [See Also](#see-also)
- [Resources](#resources)

---

## Introduction

**3D vision** extends computer vision from 2D images (H x W) to volumetric data (D x H x W) or unstructured 3D representations such as point clouds and meshes. **Medical imaging** is one of the most important application domains for 3D vision, as modalities like CT, MRI, and PET produce inherently volumetric data.

Key differences from standard 2D computer vision:

- **Dimensionality**: 3D convolutions operate over three spatial axes, increasing parameter counts and memory requirements cubically compared to 2D.
- **Anisotropic resolution**: Medical scans often have different resolutions along different axes (e.g., 0.5mm in-plane but 3mm slice thickness).
- **Domain-specific preprocessing**: Raw medical images require specialized preprocessing (windowing, resampling, bias correction) before model input.
- **Data scarcity**: Labeled 3D medical datasets are orders of magnitude smaller than ImageNet-scale 2D datasets due to annotation cost and privacy constraints.
- **Regulatory requirements**: Clinical deployment requires regulatory approval, explainability, and rigorous validation beyond standard ML benchmarks.
- **Specialized formats**: Medical images use DICOM and NIfTI formats that carry critical metadata (patient info, acquisition parameters, spatial orientation).

The field spans two broad application areas: (1) **medical imaging AI** for diagnosis, segmentation, and treatment planning, and (2) **general 3D vision** for autonomous driving, robotics, augmented reality, and 3D reconstruction.

---

## 3D Data Formats

### Volumetric Grids (Voxels)

**Voxels** (volumetric pixels) are the 3D analog of pixels. A volumetric grid is a regular 3D array of shape `(D, H, W)` where each element stores an intensity value, label, or feature vector.

- **Advantages**: Regular structure allows standard 3D convolutions; straightforward extension of 2D methods.
- **Disadvantages**: Memory scales as O(N^3); most voxels in sparse scenes are empty (wasted computation).
- **Typical sizes**: Medical CT/MRI volumes range from 128x128x64 to 512x512x512. Larger volumes require patch-based processing.

**Occupancy grids** are binary voxel grids indicating whether each cell is occupied. **Signed Distance Functions (SDF)** store the distance to the nearest surface at each voxel, with sign indicating inside/outside.

### Point Clouds

A **point cloud** is an unstructured set of 3D points `{(x_i, y_i, z_i)}` with optional per-point features (color, normal, intensity). Point clouds are the native output of LiDAR sensors and depth cameras.

- **Advantages**: Memory-efficient for sparse 3D data; preserves fine geometric detail; no discretization artifacts.
- **Disadvantages**: Unstructured and unordered, requiring specialized architectures; no regular grid for standard convolutions.
- **Typical sizes**: Indoor scenes have 10K-1M points; outdoor LiDAR sweeps have 60K-120K points; dense reconstructions can reach tens of millions.

### Meshes

A **mesh** represents 3D surfaces as a collection of **vertices** (3D points) connected by **faces** (typically triangles). Meshes are the standard representation in computer graphics and CAD.

- **Vertices**: List of (x, y, z) coordinates.
- **Faces**: List of vertex index tuples defining polygons.
- **Advantages**: Efficient surface representation; well-suited for rendering; compact for smooth surfaces.
- **Disadvantages**: Topology changes are difficult; not directly compatible with standard neural network operations.

### DICOM Format

**DICOM** (Digital Imaging and Communications in Medicine) is the universal standard for medical images. A single scan produces a **DICOM series** consisting of multiple **instances** (individual slice files), each containing:

- **Pixel data**: The actual image array.
- **Patient metadata**: Name, ID, age, sex (PHI -- Protected Health Information).
- **Acquisition parameters**: Slice thickness, pixel spacing, tube voltage, repetition time.
- **Spatial information**: Image position, image orientation, slice location.
- **Series/Study UIDs**: Unique identifiers linking slices to series and studies.

```python
import pydicom
import numpy as np
import os
from pathlib import Path


def load_dicom_series(dicom_dir):
    """Load a DICOM series from a directory into a 3D volume."""
    # Read all DICOM files in directory
    dicom_files = []
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        try:
            ds = pydicom.dcmread(fpath)
            dicom_files.append(ds)
        except Exception:
            continue  # Skip non-DICOM files

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # Sort by ImagePositionPatient (z-coordinate) or InstanceNumber
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Extract metadata
    pixel_spacing = dicom_files[0].PixelSpacing
    slice_thickness = dicom_files[0].SliceThickness
    spacing = np.array([
        float(slice_thickness),
        float(pixel_spacing[0]),
        float(pixel_spacing[1])
    ])

    # Stack slices into 3D volume
    slices = []
    for ds in dicom_files:
        # Apply RescaleSlope and RescaleIntercept to get Hounsfield Units
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixel_array = pixel_array * slope + intercept
        slices.append(pixel_array)

    volume = np.stack(slices, axis=0)  # Shape: (D, H, W)

    print(f"Volume shape: {volume.shape}")
    print(f"Spacing (D, H, W): {spacing} mm")
    print(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")

    return volume, spacing, dicom_files[0]


# Usage
# volume, spacing, ref_dcm = load_dicom_series("/path/to/dicom/series/")
```

### NIfTI Format

**NIfTI** (Neuroimaging Informatics Technology Initiative) is the standard format for neuroimaging research. Unlike DICOM (one file per slice), NIfTI stores the entire 3D (or 4D) volume in a single file.

- **File extensions**: `.nii` (uncompressed) or `.nii.gz` (gzip compressed).
- **Header**: Contains dimensions, voxel sizes, data type, and an **affine matrix** encoding the mapping from voxel indices to world coordinates (typically MNI or scanner space).
- **Affine matrix**: A 4x4 transformation matrix enabling spatial alignment across subjects.

```python
import nibabel as nib
import numpy as np


def load_nifti(filepath):
    """Load a NIfTI file and extract volume, affine, and header info."""
    nii = nib.load(filepath)

    # Get volume data as numpy array
    volume = nii.get_fdata().astype(np.float32)

    # Get affine matrix (voxel-to-world mapping)
    affine = nii.affine

    # Get header info
    header = nii.header
    voxel_sizes = header.get_zooms()  # Voxel dimensions in mm

    print(f"Volume shape: {volume.shape}")
    print(f"Voxel sizes: {voxel_sizes} mm")
    print(f"Data type: {header.get_data_dtype()}")
    print(f"Affine matrix:\n{affine}")

    return volume, affine, header


def save_nifti(volume, affine, filepath):
    """Save a numpy array as a NIfTI file."""
    nii = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(nii, filepath)
    print(f"Saved NIfTI to {filepath}")


# Usage
# volume, affine, header = load_nifti("/path/to/brain.nii.gz")
# save_nifti(processed_volume, affine, "/path/to/output.nii.gz")
```

---

## Medical Image Preprocessing

### Hounsfield Unit Windowing

In CT imaging, voxel values are measured in **Hounsfield Units (HU)**, a standardized scale where water = 0 HU and air = -1000 HU. **Windowing** maps a specific HU range to the display range [0, 1] to highlight different tissue types:

| Window Name   | Center (HU) | Width (HU) | Range (HU)       | Use Case                  |
|---------------|-------------|------------|-------------------|---------------------------|
| Soft Tissue   | 50          | 400        | [-150, 250]       | Abdomen, general          |
| Lung          | -600        | 1500       | [-1350, 150]      | Lung parenchyma           |
| Bone          | 400         | 1800       | [-500, 1300]      | Skeletal structures       |
| Brain         | 40          | 80         | [0, 80]           | Intracranial soft tissue  |
| Liver         | 60          | 160        | [-20, 140]        | Hepatic imaging           |
| Mediastinum   | 50          | 350        | [-125, 225]       | Chest mediastinum         |

Common clinical ranges (alternative specification as [min, max]):

- **Soft tissue**: [-100, 300]
- **Lung**: [-1000, 400]
- **Bone**: [-200, 2000]

```python
def apply_hu_window(volume, window_min, window_max):
    """Apply HU windowing to a CT volume."""
    volume = np.clip(volume, window_min, window_max)
    volume = (volume - window_min) / (window_max - window_min)
    return volume.astype(np.float32)


# Soft tissue window
soft_tissue = apply_hu_window(volume, -100, 300)

# Lung window
lung = apply_hu_window(volume, -1000, 400)

# Bone window
bone = apply_hu_window(volume, -200, 2000)
```

### Spacing Normalization

Medical images have **anisotropic voxel spacing** -- the physical distance between voxels differs along axes. For example, a CT scan might have 0.7mm x 0.7mm in-plane resolution but 5mm slice thickness. **Resampling to isotropic voxels** ensures uniform spatial resolution.

```python
from scipy.ndimage import zoom


def resample_volume(volume, original_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """Resample volume to target isotropic spacing."""
    original_spacing = np.array(original_spacing, dtype=np.float64)
    target_spacing = np.array(target_spacing, dtype=np.float64)

    # Compute zoom factors
    zoom_factors = original_spacing / target_spacing

    # Resample using spline interpolation
    resampled = zoom(volume, zoom_factors, order=3)  # order=3 for cubic

    print(f"Original shape: {volume.shape}, spacing: {original_spacing}")
    print(f"Resampled shape: {resampled.shape}, spacing: {target_spacing}")

    return resampled


def resample_mask(mask, original_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """Resample a segmentation mask (use nearest-neighbor interpolation)."""
    original_spacing = np.array(original_spacing, dtype=np.float64)
    target_spacing = np.array(target_spacing, dtype=np.float64)
    zoom_factors = original_spacing / target_spacing

    # order=0 for nearest-neighbor (preserves label values)
    resampled = zoom(mask, zoom_factors, order=0)
    return resampled
```

### Intensity Normalization

Different normalization strategies suit different modalities:

- **Z-score normalization**: Subtract mean, divide by standard deviation. Standard for MRI where absolute intensities are not standardized.
- **Min-max normalization**: Scale to [0, 1] range. Simple but sensitive to outliers.
- **Percentile clipping**: Clip at 0.5th and 99.5th percentiles before normalization. Robust to outlier voxels.

```python
def zscore_normalize(volume, mask=None):
    """Z-score normalization, optionally within a foreground mask."""
    if mask is not None:
        foreground = volume[mask > 0]
        mean_val = foreground.mean()
        std_val = foreground.std()
    else:
        mean_val = volume.mean()
        std_val = volume.std()

    normalized = (volume - mean_val) / (std_val + 1e-8)
    return normalized.astype(np.float32)


def percentile_clip_normalize(volume, lower=0.5, upper=99.5):
    """Clip to percentile range then normalize to [0, 1]."""
    p_low = np.percentile(volume, lower)
    p_high = np.percentile(volume, upper)
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low + 1e-8)
    return volume.astype(np.float32)
```

### Brain Extraction and Skull Stripping

**Skull stripping** removes non-brain tissue (skull, scalp, meninges) from MRI scans. This is critical for neuroimaging analyses to avoid confounds from non-brain structures.

Common tools:

- **BET** (Brain Extraction Tool, FSL): Fast, threshold-based surface evolution.
- **HD-BET**: Deep learning-based, more accurate than BET on challenging cases.
- **SynthStrip**: Robust to contrast variations, works on any MRI sequence.
- **ANTs Brain Extraction**: Template-based approach using registration.

### N4 Bias Field Correction

MRI images suffer from **bias field inhomogeneity** -- a smooth, low-frequency intensity variation across the image caused by RF coil non-uniformity. **N4ITK** (improved N3) corrects this multiplicative bias field.

```python
import SimpleITK as sitk


def n4_bias_field_correction(input_path, output_path, shrink_factor=4):
    """Apply N4 bias field correction to an MRI volume."""
    # Read image
    input_image = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # Create a brain mask (threshold-based for simplicity)
    mask_filter = sitk.OtsuThresholdImageFilter()
    mask_filter.SetInsideValue(0)
    mask_filter.SetOutsideValue(1)
    mask = mask_filter.Execute(input_image)

    # Shrink for faster processing
    shrunk_image = sitk.Shrink(
        input_image, [shrink_factor] * input_image.GetDimension()
    )
    shrunk_mask = sitk.Shrink(
        mask, [shrink_factor] * mask.GetDimension()
    )

    # N4 correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    corrector.SetConvergenceThreshold(0.001)
    corrected = corrector.Execute(shrunk_image, shrunk_mask)

    # Get the bias field and apply to full-resolution image
    log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
    corrected_full = input_image / sitk.Exp(log_bias_field)

    sitk.WriteImage(corrected_full, output_path)
    print(f"N4 corrected image saved to {output_path}")
    return corrected_full
```

### Registration and Atlas Alignment

**Image registration** aligns images from different subjects, time points, or modalities into a common coordinate system. This enables voxel-wise comparisons and atlas-based analyses.

- **Rigid registration**: 6 degrees of freedom (3 translation + 3 rotation). Used for intra-subject alignment.
- **Affine registration**: 12 degrees of freedom (rigid + scaling + shearing). Corrects for global shape differences.
- **Deformable registration**: Non-linear, dense displacement fields. Captures local anatomical variability.
- **Atlas-based**: Align to a standard template (e.g., MNI152 for brain). Enables group analyses.

Tools: **ANTs** (SyN algorithm, gold standard for deformable), **FSL FLIRT/FNIRT**, **SimpleITK**, **VoxelMorph** (deep learning-based).

**Complete CT preprocessing pipeline:**

```python
import numpy as np
from scipy.ndimage import zoom


def ct_preprocessing_pipeline(volume, spacing, target_spacing=(1.0, 1.0, 1.0),
                               hu_window=(-100, 300)):
    """
    Complete CT preprocessing pipeline.

    Steps:
    1. Resample to isotropic spacing
    2. Apply HU windowing
    3. Normalize to [0, 1]
    """
    # Step 1: Resample to target spacing
    zoom_factors = np.array(spacing) / np.array(target_spacing)
    volume_resampled = zoom(volume, zoom_factors, order=3)

    # Step 2: HU windowing
    hu_min, hu_max = hu_window
    volume_windowed = np.clip(volume_resampled, hu_min, hu_max)

    # Step 3: Normalize to [0, 1]
    volume_normalized = (volume_windowed - hu_min) / (hu_max - hu_min)

    print(f"CT pipeline: {volume.shape} -> {volume_normalized.shape}")
    print(f"Spacing: {spacing} -> {target_spacing}")
    print(f"HU window: [{hu_min}, {hu_max}]")

    return volume_normalized.astype(np.float32)


def ct_preprocessing_with_mask(volume, mask, spacing,
                                target_spacing=(1.0, 1.0, 1.0),
                                hu_window=(-100, 300)):
    """CT preprocessing with corresponding segmentation mask."""
    zoom_factors = np.array(spacing) / np.array(target_spacing)

    volume_resampled = zoom(volume, zoom_factors, order=3)
    mask_resampled = zoom(mask, zoom_factors, order=0)  # Nearest for labels

    hu_min, hu_max = hu_window
    volume_windowed = np.clip(volume_resampled, hu_min, hu_max)
    volume_normalized = (volume_windowed - hu_min) / (hu_max - hu_min)

    return volume_normalized.astype(np.float32), mask_resampled.astype(np.int64)
```

**MRI preprocessing pipeline:**

```python
import SimpleITK as sitk
import numpy as np


def mri_preprocessing_pipeline(input_path, output_path,
                                target_spacing=(1.0, 1.0, 1.0),
                                do_bias_correction=True,
                                do_skull_strip=False):
    """
    Complete MRI preprocessing pipeline.

    Steps:
    1. N4 bias field correction
    2. Resample to isotropic spacing
    3. Z-score intensity normalization
    """
    image = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # Step 1: N4 bias field correction
    if do_bias_correction:
        mask_filter = sitk.OtsuThresholdImageFilter()
        mask_filter.SetInsideValue(0)
        mask_filter.SetOutsideValue(1)
        mask = mask_filter.Execute(image)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        image = corrector.Execute(image, mask)

    # Step 2: Resample to isotropic spacing
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkBSpline)
    image = resampler.Execute(image)

    # Step 3: Z-score normalization
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    # Compute stats on foreground (non-zero voxels)
    foreground = volume[volume > 0]
    if len(foreground) > 0:
        mean_val = foreground.mean()
        std_val = foreground.std()
        volume = (volume - mean_val) / (std_val + 1e-8)

    # Convert back to SimpleITK and save
    result = sitk.GetImageFromArray(volume)
    result.CopyInformation(image)
    sitk.WriteImage(result, output_path)

    print(f"MRI pipeline complete: {output_path}")
    return result
```

---

## 3D CNN Architectures

### 3D Convolutions

A **3D convolution** extends the 2D convolution by adding a depth dimension to the kernel. For a kernel of size `(k_d, k_h, k_w)`, the operation slides across three spatial dimensions.

```
Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
```

- **Parameters**: For a 3x3x3 kernel with C_in input channels and C_out output channels: `C_out x C_in x 3 x 3 x 3` weights.
- **Computation**: A single 3D conv layer with 64 input and 64 output channels using 3x3x3 kernels has 64 x 64 x 27 = 110,592 parameters (plus bias).
- **Memory**: Feature maps scale cubically with spatial dimensions. A 128x128x128 volume with 64 channels requires 128 MB in float32.

### 3D ResNet

**3D ResNet** extends ResNet to volumetric data by replacing all 2D operations with 3D counterparts.

```python
import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """3D residual block with two 3x3x3 convolutions."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet3D(nn.Module):
    """3D ResNet for volumetric classification."""

    def __init__(self, in_channels=1, num_classes=2,
                 channels=(64, 128, 256, 512), blocks_per_stage=(2, 2, 2, 2)):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Residual stages
        self.stages = nn.ModuleList()
        in_ch = channels[0]
        for i, (out_ch, num_blocks) in enumerate(
            zip(channels, blocks_per_stage)
        ):
            stride = 1 if i == 0 else 2
            blocks = [ResBlock3D(in_ch, out_ch, stride=stride)]
            for _ in range(1, num_blocks):
                blocks.append(ResBlock3D(out_ch, out_ch))
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x).flatten(1)
        return self.fc(x)


# Usage
model = ResNet3D(in_channels=1, num_classes=3)
x = torch.randn(2, 1, 64, 64, 64)  # (B, C, D, H, W)
out = model(x)
print(f"Output shape: {out.shape}")  # (2, 3)
```

### 3D DenseNet

**3D DenseNet** uses dense connections within each block, where every layer receives feature maps from all preceding layers. This promotes feature reuse and reduces the number of parameters needed compared to ResNet.

Key differences from 3D ResNet:
- Each layer outputs `k` feature maps (the **growth rate**), and all are concatenated.
- **Transition layers** between blocks use 1x1x1 convolution and 3D average pooling to reduce dimensions.
- More parameter-efficient but memory-intensive due to concatenation.

### SE-ResNet3D

**Squeeze-and-Excitation (SE)** blocks add channel attention to 3D ResNets. The SE block:

1. **Squeeze**: Global average pooling over spatial dimensions to produce a channel descriptor.
2. **Excitation**: Two FC layers (reduce then expand) with sigmoid to produce per-channel weights.
3. **Scale**: Multiply original feature maps by channel weights.

```python
class SE3DBlock(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1, 1)
        return x * w
```

### I3D Inflated 3D ConvNets

**I3D** (Inflated Inception-V1) converts a pretrained 2D CNN to 3D by "inflating" 2D kernels:

- A 2D kernel of shape `(k, k)` becomes `(k, k, k)` in 3D.
- Pretrained 2D weights are replicated along the temporal/depth dimension and divided by `k` to preserve output magnitude.
- This provides a strong initialization for 3D networks using ImageNet-pretrained features.

Inflation procedure: For a 2D weight tensor `W` of shape `(C_out, C_in, k, k)`, create `W_3d` of shape `(C_out, C_in, k, k, k)` where each depth slice equals `W / k`.

### C3D for Video Understanding

**C3D** (Convolutional 3D) is an early architecture for spatiotemporal feature learning from video:

- All convolutions are 3x3x3 with stride 1, padding 1.
- All pooling layers are 2x2x2 except the first (1x2x2 to preserve temporal resolution early).
- 8 convolution layers, 5 pooling layers, 2 FC layers.
- Input: 16 frames of 112x112 pixels.
- Pretrained on Sports-1M dataset.

### Memory Considerations

3D convolutions are significantly more memory-intensive than 2D:

| Aspect          | 2D (256x256)    | 3D (128x128x128)     |
|-----------------|------------------|-----------------------|
| Input voxels    | 65,536           | 2,097,152             |
| 3x3 kernel      | 9 weights/ch     | 27 weights/ch (3x3x3) |
| Feature map (64ch) | 16 MB         | 512 MB                |

**Strategies to manage memory:**

- **Patch-based training**: Train on random 3D patches (e.g., 96x96x96) rather than full volumes.
- **Mixed precision** (FP16): Halves memory with minimal accuracy loss.
- **Gradient checkpointing**: Trade compute for memory by recomputing activations during backward pass.
- **Sparse convolutions**: Only compute on occupied voxels (MinkowskiEngine, SpConv).

**Loading pretrained 3D models:**

```python
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


def load_pretrained_3d_resnet(num_classes=10, pretrained=True):
    """Load a pretrained 3D ResNet-18 and adapt for custom classes."""
    if pretrained:
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    else:
        model = r3d_18(weights=None)

    # Replace the final FC layer for custom number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# For single-channel medical images, modify the first conv layer
def adapt_3d_resnet_for_medical(num_classes=2):
    """Adapt pretrained 3D ResNet for single-channel medical volumes."""
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    # Modify first conv: average RGB weights for single channel
    old_conv = model.stem[0]
    new_conv = nn.Conv3d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )
    # Average pretrained RGB weights across channel dimension
    new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    model.stem[0] = new_conv

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
```

---

## 3D U-Net for Medical Segmentation

### Standard 3D U-Net

**3D U-Net** extends the U-Net architecture to volumetric data. It follows the encoder-decoder design with skip connections:

- **Encoder**: Series of 3D conv blocks + 3D max pooling for downsampling.
- **Bottleneck**: Deepest layer with highest channel count.
- **Decoder**: 3D transposed convolution (or trilinear upsampling) + skip connections from encoder.
- **Skip connections**: Concatenate encoder features with decoder features at matching resolution.

```python
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Double 3D convolution block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation."""

    def __init__(self, in_channels=1, num_classes=2,
                 features=(32, 64, 128, 256)):
        super().__init__()

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock3D(ch, f))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch = f

        # Bottleneck
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        ch = features[-1] * 2
        for f in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose3d(ch, f, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock3D(f * 2, f))  # f*2 due to skip concat
            ch = f

        # Output
        self.final_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, skip_connections
        ):
            x = upconv(x)
            # Handle shape mismatches from odd dimensions
            if x.shape != skip.shape:
                x = nn.functional.pad(x, [
                    0, skip.shape[4] - x.shape[4],
                    0, skip.shape[3] - x.shape[3],
                    0, skip.shape[2] - x.shape[2],
                ])
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.final_conv(x)


# Usage
model = UNet3D(in_channels=1, num_classes=4)
x = torch.randn(1, 1, 64, 64, 64)
out = model(x)
print(f"Output shape: {out.shape}")  # (1, 4, 64, 64, 64)
```

### V-Net

**V-Net** introduced several improvements over 3D U-Net:

- **Residual connections** within each stage (instead of plain convolutions).
- **Dice loss** as the training objective, directly optimizing the overlap metric.
- **PReLU** activation instead of ReLU.
- Designed for prostate segmentation in MRI but widely adopted.

**Dice loss**:

```python
def dice_loss(pred, target, smooth=1.0):
    """Dice loss for binary or multi-class segmentation."""
    pred = torch.sigmoid(pred)
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()
```

### Attention U-Net

**Attention U-Net** adds **attention gates** to the skip connections. Instead of blindly concatenating encoder features, attention gates learn to highlight relevant regions and suppress irrelevant ones.

The attention gate computes:
1. Linearly project both the gating signal (from decoder) and the skip connection (from encoder).
2. Sum and apply ReLU.
3. Apply a 1x1x1 convolution followed by sigmoid to produce an attention map.
4. Multiply the skip connection by the attention map.

```python
class AttentionGate3D(nn.Module):
    """Attention gate for 3D U-Net skip connections."""

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.W_skip = nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        # Upsample gate to match skip spatial size if needed
        if g.shape[2:] != s.shape[2:]:
            g = nn.functional.interpolate(g, size=s.shape[2:], mode="trilinear",
                                          align_corners=False)
        attention = self.psi(self.relu(g + s))
        return skip * attention
```

### nnU-Net

**nnU-Net** (no-new-Net) is a self-configuring segmentation framework that automatically adapts preprocessing, architecture, and training to any new dataset. It has won or placed highly in the majority of medical segmentation challenges since its introduction.

Key principles:

- **Automatic preprocessing**: Analyzes dataset statistics to determine resampling strategy, normalization, and patch size.
- **Three configurations**: 2D (slice-by-slice), 3D full-resolution, and 3D cascade (low-res followed by high-res refinement).
- **Architecture adaptation**: Automatically selects encoder depth, feature map sizes, batch size based on available GPU memory and dataset properties.
- **Robust training**: 5-fold cross-validation, extensive data augmentation, combined Dice + cross-entropy loss.
- **Postprocessing**: Automatic selection of best configuration and ensemble strategies.

```python
# nnU-Net v2 usage (command-line based)
# Step 1: Set environment variables
# export nnUNet_raw="/path/to/nnUNet_raw"
# export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
# export nnUNet_results="/path/to/nnUNet_results"

# Step 2: Organize data in nnU-Net format
# nnUNet_raw/Dataset001_BrainTumor/
#   imagesTr/    (training images: case_0000.nii.gz)
#   labelsTr/    (training labels: case.nii.gz)
#   imagesTs/    (test images)
#   dataset.json

# Step 3: Plan and preprocess
# nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# Step 4: Train (all 5 folds, 3d_fullres)
# nnUNetv2_train 1 3d_fullres 0  # fold 0
# nnUNetv2_train 1 3d_fullres 1  # fold 1
# ... repeat for folds 2, 3, 4

# Step 5: Predict
# nnUNetv2_predict -i /path/to/test -o /path/to/output -d 1 \
#   -c 3d_fullres -f 0 1 2 3 4

# Creating dataset.json programmatically:
import json

dataset_json = {
    "channel_names": {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    },
    "labels": {
        "background": 0,
        "edema": 1,
        "enhancing_tumor": 2,
        "necrosis": 3
    },
    "numTraining": 484,
    "file_ending": ".nii.gz"
}

with open("dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)
```

### MONAI Framework

**MONAI** (Medical Open Network for AI) is a PyTorch-based framework specifically designed for medical imaging deep learning. It provides:

- **Transforms**: Medical-specific data augmentation and preprocessing.
- **Networks**: Pretrained and configurable architectures (UNet, UNETR, SwinUNETR, SegResNet).
- **Losses**: Dice, DiceCE, Focal, Tversky, and more.
- **Metrics**: Dice score, Hausdorff distance, surface distance.
- **Data loading**: Support for DICOM, NIfTI, and other medical formats.

```python
import torch
import monai
from monai.networks.nets import UNet, SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference


# Define transforms for training
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"],
             pixdim=(1.0, 1.0, 1.0),
             mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"],
                         a_min=-100, a_max=300,
                         b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1, neg=1, num_samples=4,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
    EnsureTyped(keys=["image", "label"]),
])

# Create model
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=4,
    feature_size=48,
    use_checkpoint=True,  # Gradient checkpointing for memory
)

# Loss and optimizer
loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Sliding window inference for full-volume prediction
def predict_full_volume(model, image, device):
    """Predict on a full volume using sliding window."""
    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=image.to(device),
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
        )
    return output.argmax(dim=1)
```

---

## Point Cloud Networks

### PointNet

**PointNet** (Qi et al., 2017) is the foundational architecture for direct point cloud processing. Key design principles:

- **Permutation invariance**: Point clouds are unordered sets. PointNet processes each point independently through shared MLPs, then applies a symmetric function (max pooling) to aggregate.
- **Spatial Transformer Network**: Learns a 3x3 (and 64x64) transformation matrix to align the input, providing invariance to rigid transformations.
- **Architecture**: Input (N x 3) -> T-Net -> shared MLP (64, 128, 1024) -> max pool -> MLP classifier.

**Limitations**: PointNet treats each point independently before max pooling, so it cannot capture local geometric structures.

### PointNet++

**PointNet++** (Qi et al., 2017) introduces hierarchical feature learning:

1. **Sampling**: Select a subset of points as centroids using Farthest Point Sampling (FPS).
2. **Grouping**: Find neighbors within a radius (ball query) around each centroid.
3. **PointNet**: Apply PointNet to each local neighborhood.
4. **Repeat**: Stack multiple Set Abstraction (SA) layers for hierarchical features.

Multi-scale grouping (MSG) applies PointNet at multiple radii to handle varying point density.

### DGCNN

**Dynamic Graph CNN** (Wang et al., 2019) constructs a k-NN graph in feature space and applies edge convolutions:

- **EdgeConv**: For each point, compute edge features relative to its k nearest neighbors, apply MLP, and aggregate.
- **Dynamic**: The k-NN graph is recomputed at each layer based on updated features (not just spatial coordinates).
- Combines local geometric structure with global feature learning.

### Point Transformer

**Point Transformer** (Zhao et al., 2021) applies self-attention to point clouds:

- **Vector attention**: Attention weights are vectors (not scalars), allowing channel-wise modulation.
- **Local attention**: Computed within k-NN neighborhoods for efficiency.
- **Positional encoding**: Relative position encodings capture geometric relationships.
- Achieves state-of-the-art on multiple point cloud benchmarks.

### PointPillars

**PointPillars** (Lang et al., 2019) converts point clouds to a 2D pseudo-image for fast detection:

1. **Pillar creation**: Discretize the x-y plane into a grid. Points in each grid cell form a "pillar."
2. **Pillar features**: Apply PointNet to each pillar to produce a fixed-size feature vector.
3. **Scatter**: Place pillar features into a 2D grid (bird's eye view pseudo-image).
4. **2D backbone**: Apply standard 2D CNN detection head (SSD-style).

This approach achieves real-time performance (62 Hz) with competitive accuracy.

**PointNet classification in PyTorch:**

```python
import torch
import torch.nn as nn


class TNet(nn.Module):
    """Spatial Transformer Network for PointNet."""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        batch_size = x.size(0)
        # x: (B, k, N)
        out = self.mlp(x)
        out = out.max(dim=2)[0]  # (B, 1024)
        out = self.fc(out)  # (B, k*k)
        # Initialize as identity
        identity = torch.eye(self.k, device=x.device).flatten().unsqueeze(0)
        out = out + identity
        return out.view(batch_size, self.k, self.k)


class PointNetClassifier(nn.Module):
    """PointNet classification network."""

    def __init__(self, num_classes=40, num_points=1024):
        super().__init__()
        self.num_points = num_points

        # Input transform
        self.input_transform = TNet(k=3)

        # MLP 1
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Feature transform
        self.feature_transform = TNet(k=64)

        # MLP 2
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud
        Returns:
            logits: (B, num_classes)
        """
        batch_size = x.size(0)
        x = x.transpose(1, 2)  # (B, 3, N)

        # Input transform
        t1 = self.input_transform(x)  # (B, 3, 3)
        x = torch.bmm(t1, x)  # (B, 3, N)

        # MLP 1
        x = self.mlp1(x)  # (B, 64, N)

        # Feature transform
        t2 = self.feature_transform(x)  # (B, 64, 64)
        x = torch.bmm(t2, x)  # (B, 64, N)

        # MLP 2
        x = self.mlp2(x)  # (B, 1024, N)

        # Max pooling (symmetric function for permutation invariance)
        x = x.max(dim=2)[0]  # (B, 1024)

        # Classification
        logits = self.classifier(x)  # (B, num_classes)

        return logits, t2  # Return t2 for regularization loss


def pointnet_regularization_loss(transform, weight=0.001):
    """Regularization loss to keep feature transform close to orthogonal."""
    batch_size = transform.size(0)
    k = transform.size(1)
    identity = torch.eye(k, device=transform.device).unsqueeze(0).expand(
        batch_size, -1, -1
    )
    diff = torch.bmm(transform, transform.transpose(1, 2)) - identity
    return weight * diff.norm(dim=(1, 2)).mean()


# Usage
model = PointNetClassifier(num_classes=40, num_points=1024)
points = torch.randn(8, 1024, 3)  # (B, N, 3)
logits, feat_transform = model(points)
print(f"Logits shape: {logits.shape}")  # (8, 40)

# Combined loss
cls_loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 40, (8,)))
reg_loss = pointnet_regularization_loss(feat_transform)
total_loss = cls_loss + reg_loss
```

---

## 3D Object Detection

### VoxelNet

**VoxelNet** (Zhou and Tuber, 2018) is an end-to-end 3D detection network:

1. **Voxelization**: Divide the 3D space into a regular voxel grid. Group points into voxels.
2. **Voxel Feature Encoding (VFE)**: Apply stacked PointNet-like layers within each voxel to produce voxel-wise features.
3. **3D Convolutions**: Apply sparse 3D convolutions on the voxel grid.
4. **RPN**: Region Proposal Network on the bird's eye view feature map.

Limitation: 3D convolutions are computationally expensive. SECOND and PointPillars addressed this.

### PointPillars for Detection

PointPillars replaces voxels with **pillars** (vertical columns) and eliminates 3D convolutions entirely:

- Input point cloud is discretized into a 2D x-y grid.
- Each pillar's points are encoded with a simplified PointNet.
- Results in a 2D pseudo-image processed by a standard 2D backbone.
- Achieves 62 Hz inference on KITTI benchmark -- fast enough for real-time autonomous driving.

### CenterPoint

**CenterPoint** (Yin et al., 2021) is a center-based 3D object detector:

1. **Backbone**: VoxelNet or PointPillars-style feature extraction.
2. **Center heatmap**: Predict a heatmap of object centers in bird's eye view.
3. **Per-center regression**: For each detected center, regress 3D box dimensions, z-center, and orientation.
4. **Two-stage refinement**: Optional second stage refines detections using point features.

Advantages: Anchor-free design avoids complex anchor tuning. Naturally handles arbitrary orientations.

### SECOND

**SECOND** (Sparsely Embedded Convolutional Detection) introduces **sparse convolutions** for 3D detection:

- Only computes convolutions at occupied voxel locations.
- Uses a hash table to track active voxels.
- 3-5x faster than dense 3D convolutions for typical LiDAR scenes (which are 95%+ empty).
- Implementation: `spconv` library provides efficient CUDA kernels for sparse 3D convolutions.

### BEV Methods

**Bird's Eye View (BEV)** methods project 3D information onto a top-down 2D plane:

- **BEVFusion**: Fuses LiDAR BEV features with camera features projected to BEV.
- **BEVDet**: Camera-only 3D detection by lifting 2D image features to BEV using depth estimation.
- **BEVFormer**: Transformer-based BEV generation from multi-view cameras.

BEV is natural for autonomous driving because the ground plane captures the most relevant spatial relationships for driving decisions.

**3D detection with MMDetection3D:**

```python
# 3D object detection with MMDetection3D
# Installation: pip install mmdet3d

# Example configuration for PointPillars on KITTI
# File: pointpillars_kitti.py (MMDetection3D config)

# Command-line usage:
# python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6_kitti-3d-3class.py

# Inference example:
from mmdet3d.apis import init_model, inference_detector

config_file = "configs/pointpillars/pointpillars_hv_secfpn_8xb6_kitti-3d-3class.py"
checkpoint_file = "checkpoints/pointpillars_kitti.pth"

# Build model
model = init_model(config_file, checkpoint_file, device="cuda:0")

# Run inference on a point cloud
result = inference_detector(model, "demo/data/kitti/000008.bin")

# result contains:
# - pred_instances_3d.bboxes_3d: 3D bounding boxes (x, y, z, w, l, h, yaw)
# - pred_instances_3d.scores_3d: confidence scores
# - pred_instances_3d.labels_3d: class labels

for i in range(len(result.pred_instances_3d.scores_3d)):
    score = result.pred_instances_3d.scores_3d[i].item()
    if score > 0.3:
        bbox = result.pred_instances_3d.bboxes_3d[i]
        label = result.pred_instances_3d.labels_3d[i].item()
        print(f"Class {label}, Score {score:.2f}, Box: {bbox}")
```

---

## Medical Image Classification

### Patch-Based Classification

For large 3D volumes that do not fit in GPU memory, **patch-based classification** extracts regions of interest (ROIs) and classifies each patch:

1. **Extract patches**: Slide a window across the volume or sample around candidate locations.
2. **Classify patches**: Apply a 3D CNN to each patch.
3. **Aggregate**: Combine patch-level predictions via majority voting, averaging, or learned aggregation.

This approach is common in pathology (whole-slide images) and CT nodule classification.

### Multi-Instance Learning

**Multi-Instance Learning (MIL)** treats each patient/scan as a **bag** of instances (patches):

- **Bag label**: The scan-level label (e.g., cancer present / absent).
- **Instance labels**: Unknown -- only the bag label is provided.
- **Key assumption**: A positive bag contains at least one positive instance; a negative bag has no positive instances.

MIL approaches:

- **Attention-based MIL**: Learn attention weights for each instance, then aggregate with weighted sum.
- **Max-pooling MIL**: The bag prediction is the maximum instance prediction.
- **Transformer MIL**: Use self-attention across instances.

### Transfer Learning 2D to 3D

When 3D pretrained models are unavailable, **2D-to-3D transfer learning** leverages ImageNet-pretrained 2D models:

1. **Slice-level**: Process each 2D slice with a pretrained 2D CNN to extract features.
2. **Aggregation**: Combine slice features using RNN, attention, or max pooling.
3. **Fine-tuning**: End-to-end fine-tuning on the 3D task.

Alternative strategies:
- **Multi-planar**: Extract axial, coronal, sagittal slices and fuse predictions from three 2D networks.
- **2.5D approach**: Use three adjacent slices as RGB channels for a pretrained 2D model.
- **Weight inflation**: Replicate 2D weights along the depth dimension (I3D approach).

### Class Imbalance Handling

Medical datasets are typically severely imbalanced (e.g., 1% positive lesion voxels):

- **Oversampling**: Ensure each batch has balanced positive/negative examples (MONAI's `RandCropByPosNegLabeld`).
- **Loss weighting**: Inverse frequency weighting in cross-entropy loss.
- **Focal loss**: Down-weight easy examples, focus on hard ones. `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`.
- **Dice loss**: Directly optimizes the overlap metric, inherently handles imbalance.
- **Combined losses**: DiceCE (Dice + Cross-Entropy) is the standard in medical segmentation.

**Medical image classifier with MONAI:**

```python
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, Resized, RandFlipd, RandRotate90d,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader


def build_medical_classifier():
    """Build a medical image classifier using MONAI DenseNet121."""
    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
    )
    return model


def get_classification_transforms(phase="train"):
    """Get transforms for medical image classification."""
    common = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"],
                             a_min=-100, a_max=300,
                             b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=["image"], spatial_size=(128, 128, 64)),
    ]
    if phase == "train":
        common.extend([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
        ])
    common.append(EnsureTyped(keys=["image"]))
    return Compose(common)


def train_medical_classifier(train_files, val_files, num_epochs=100):
    """Training loop for medical image classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CacheDataset(train_files, get_classification_transforms("train"))
    val_ds = CacheDataset(val_files, get_classification_transforms("val"))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4)

    model = build_medical_classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_classifier.pth")

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Acc: {val_acc:.4f}")
```

---

## Radiomics and Feature Extraction

### PyRadiomics

**Radiomics** extracts quantitative features from medical images that capture shape, texture, and intensity information beyond what the human eye can discern. **PyRadiomics** is the standard open-source library for radiomics feature extraction.

Feature categories:

1. **First-order statistics**: Mean, median, standard deviation, skewness, kurtosis, entropy, energy, uniformity (18 features).
2. **Shape features**: Volume, surface area, sphericity, elongation, flatness, maximum 3D diameter (14 features).
3. **Texture features (GLCM)**: Gray-Level Co-occurrence Matrix features -- contrast, correlation, energy, homogeneity, entropy (24 features).
4. **Texture features (GLRLM)**: Gray-Level Run-Length Matrix -- short/long run emphasis, run percentage (16 features).
5. **Texture features (GLSZM)**: Gray-Level Size Zone Matrix -- small/large area emphasis, zone percentage (16 features).
6. **Texture features (GLDM)**: Gray-Level Dependence Matrix (14 features).
7. **Texture features (NGTDM)**: Neighborhood Gray-Tone Difference Matrix -- coarseness, contrast, busyness (5 features).

### Texture Features

**GLCM** (Gray-Level Co-occurrence Matrix) captures how often pairs of voxel values occur at specified spatial offsets:

- **Contrast**: Measures local intensity variation.
- **Correlation**: Measures linear dependency between neighboring voxels.
- **Energy** (Angular Second Moment): Measures uniformity of the distribution.
- **Homogeneity**: Measures closeness to the diagonal (similar neighbor values).

**GLRLM** (Gray-Level Run-Length Matrix) describes consecutive voxels with the same intensity in a given direction:

- **Short Run Emphasis (SRE)**: Dominated by short runs (fine texture).
- **Long Run Emphasis (LRE)**: Dominated by long runs (coarse texture).

### Shape and First-Order Features

**Shape features** describe the geometric properties of the segmented region:

- **Volume**: Total number of voxels times voxel volume.
- **Surface area**: Computed from the mesh of the segmentation boundary.
- **Sphericity**: How close the shape is to a sphere. `Sphericity = (36 * pi * V^2)^(1/3) / A` where V is volume and A is surface area.
- **Elongation**: Ratio of second-largest to largest principal axis length.

**First-order features** describe the distribution of voxel intensities within the ROI without considering spatial relationships.

**Radiomics extraction pipeline:**

```python
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def extract_radiomics_features(image_path, mask_path, params_path=None):
    """
    Extract radiomics features from an image and mask.

    Args:
        image_path: Path to the image (NIfTI or DICOM).
        mask_path: Path to the segmentation mask.
        params_path: Optional path to PyRadiomics parameter file (YAML).

    Returns:
        Dictionary of feature names and values.
    """
    if params_path:
        extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
        # Enable all feature classes
        extractor.enableAllFeatures()
        # Optional: enable specific image types
        extractor.enableImageTypeByName("Original")
        extractor.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 3.0, 5.0]})
        extractor.enableImageTypeByName("Wavelet")

    # Extract features
    result = extractor.execute(image_path, mask_path)

    # Filter out diagnostics, keep only features
    features = {}
    for key, value in result.items():
        if not key.startswith("diagnostics_"):
            features[key] = float(value)

    print(f"Extracted {len(features)} features")
    return features


def radiomics_ml_pipeline(image_paths, mask_paths, labels, n_top_features=20):
    """
    Complete radiomics + ML pipeline for clinical prediction.

    Args:
        image_paths: List of image file paths.
        mask_paths: List of mask file paths.
        labels: Binary labels (0 or 1).
        n_top_features: Number of top features to select.
    """
    # Step 1: Extract features for all subjects
    all_features = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        features = extract_radiomics_features(img_path, mask_path)
        all_features.append(features)

    # Convert to DataFrame
    df_features = pd.DataFrame(all_features)
    print(f"Feature matrix shape: {df_features.shape}")

    # Step 2: Remove features with zero variance
    variance = df_features.var()
    df_features = df_features.loc[:, variance > 0]
    print(f"After variance filter: {df_features.shape}")

    # Step 3: Handle missing/infinite values
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(df_features.median())

    # Step 4: Build ML pipeline
    X = df_features.values
    y = np.array(labels)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(mutual_info_classif, k=n_top_features)),
        ("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )),
    ])

    # Step 5: Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    print(f"Cross-validation AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Step 6: Fit on all data and get selected features
    pipeline.fit(X, y)
    selector = pipeline.named_steps["selector"]
    selected_mask = selector.get_support()
    selected_features = df_features.columns[selected_mask].tolist()
    print(f"Top {n_top_features} features: {selected_features}")

    return pipeline, selected_features, scores
```

---

## Neural Radiance Fields

### NeRF Architecture

**Neural Radiance Fields (NeRF)** represent a 3D scene as a continuous function that maps a 5D input (3D position + 2D viewing direction) to color and volume density:

```
F: (x, y, z, theta, phi) -> (r, g, b, sigma)
```

- **(x, y, z)**: 3D spatial position.
- **(theta, phi)**: Viewing direction (allows view-dependent effects like specular highlights).
- **(r, g, b)**: Emitted color.
- **sigma**: Volume density (opacity).

The function `F` is parameterized by an MLP (typically 8 layers, 256 channels). The density `sigma` depends only on position (view-independent), while color depends on both position and direction.

**Volume rendering**: To render a pixel, cast a ray from the camera through the pixel. Sample points along the ray. For each sample point, query the MLP for color and density. Accumulate color using the volume rendering integral:

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i
```

where `T_i = exp(-sum_{j<i} sigma_j * delta_j)` is the accumulated transmittance and `delta_i` is the distance between consecutive samples.

### Positional Encoding

Raw (x, y, z) coordinates are low-dimensional and cannot represent high-frequency details. **Positional encoding** maps low-dimensional inputs to a higher-dimensional space using sinusoidal functions:

```
gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p),
            sin(2^1 * pi * p), cos(2^1 * pi * p),
            ...,
            sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]
```

For position coordinates, L=10 (maps 3D to 60D). For direction, L=4 (maps 2D to 24D). This enables the MLP to learn high-frequency scene details.

### Instant-NGP

**Instant-NGP** (Mueller et al., 2022) replaces positional encoding with a **multi-resolution hash encoding** for dramatically faster training (seconds to minutes instead of hours):

- **Hash grid**: Multiple levels of 3D hash grids at different resolutions. Each grid vertex stores a learnable feature vector.
- **Trilinear interpolation**: For a query point, interpolate features from surrounding grid vertices at each resolution level.
- **Concatenation**: Concatenate features from all levels and pass through a small MLP.
- **Hash collisions**: Handled implicitly -- the optimization resolves collisions through gradient-based learning.

Training speedup: approximately 1000x faster than original NeRF. Can train a scene in 5-15 seconds on modern GPUs.

### 3D Gaussian Splatting

**3D Gaussian Splatting** (Kerbl et al., 2023) represents scenes as a set of explicit 3D Gaussians rather than an implicit neural field:

- Each Gaussian has: position (mean), covariance (3x3 matrix parameterized as rotation + scale), opacity, and spherical harmonics coefficients for view-dependent color.
- **Rendering**: Project 3D Gaussians onto the image plane (splatting), sort by depth, and alpha-blend front-to-back.
- **Advantages over NeRF**: Real-time rendering (100+ FPS), explicit representation allows easy editing, faster training.
- **Optimization**: Initialize from sparse SfM point cloud. Optimize all Gaussian parameters via gradient descent with differentiable splatting.

Applications for both NeRF and Gaussian Splatting:

- **Novel view synthesis**: Render the scene from any camera position.
- **3D reconstruction**: Extract geometry from the learned representation.
- **Scene editing**: Modify, add, or remove objects.
- **Digital twins**: Create photorealistic digital copies of real environments.
- **Medical imaging**: Reconstruct 3D anatomy from sparse 2D X-rays or limited-angle CT.

---

## Data Augmentation for 3D and Medical Imaging

Data augmentation is critical for 3D medical imaging due to small dataset sizes. However, augmentations must be carefully chosen to preserve anatomical plausibility.

**Spatial augmentations:**

- **3D rotation**: Small angles (e.g., +/- 15 degrees) to avoid unrealistic anatomy. Full 360-degree rotation is appropriate for non-anatomical data like point clouds.
- **3D elastic deformation**: Smooth random displacement fields simulate anatomical variability. Key parameters: deformation magnitude and smoothing sigma.
- **Random cropping**: Extract random sub-volumes. Use `RandCropByPosNegLabeld` in MONAI to ensure foreground coverage.
- **Random scaling**: Slight zoom in/out (0.9x to 1.1x) simulates size variability.
- **Flipping**: Left-right flipping is usually safe; superior-inferior flipping is generally not appropriate for medical images.

**Intensity augmentations:**

- **Brightness/contrast**: Random additive offset and multiplicative scaling.
- **Gaussian noise**: Simulate acquisition noise.
- **Gaussian blur/sharpen**: Simulate varying acquisition quality.
- **Gamma correction**: Non-linear intensity transformation.

**Advanced augmentations:**

- **MixUp for 3D**: Linear interpolation between two volumes and their labels.
- **CutMix for 3D**: Replace a random cubic region with content from another volume.
- **Simulation-based**: Simulate different scanner parameters, slice thicknesses, or reconstruction kernels.

**3D augmentation pipeline:**

```python
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandAffined,
    RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandShiftIntensityd,
    RandAdjustContrastd, RandCropByPosNegLabeld,
    EnsureTyped,
)


def elastic_deformation_3d(volume, alpha=15, sigma=3, random_state=None):
    """
    Apply random elastic deformation to a 3D volume.

    Args:
        volume: 3D numpy array (D, H, W).
        alpha: Deformation magnitude.
        sigma: Smoothing sigma (controls smoothness).
    """
    if random_state is None:
        random_state = np.random.RandomState()

    shape = volume.shape
    # Generate random displacement fields
    dx = gaussian_filter(random_state.randn(*shape) * alpha, sigma, mode="constant")
    dy = gaussian_filter(random_state.randn(*shape) * alpha, sigma, mode="constant")
    dz = gaussian_filter(random_state.randn(*shape) * alpha, sigma, mode="constant")

    # Create coordinate grids
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij"
    )

    # Apply displacement
    indices = [
        np.clip(z + dz, 0, shape[0] - 1),
        np.clip(y + dy, 0, shape[1] - 1),
        np.clip(x + dx, 0, shape[2] - 1),
    ]

    return map_coordinates(volume, indices, order=1, mode="reflect")


def mixup_3d(volume1, label1, volume2, label2, alpha=0.2):
    """MixUp augmentation for 3D volumes."""
    lam = np.random.beta(alpha, alpha)
    mixed_volume = lam * volume1 + (1 - lam) * volume2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_volume, mixed_label


def get_monai_augmentation_pipeline(patch_size=(96, 96, 96)):
    """Comprehensive MONAI augmentation pipeline for 3D medical images."""
    return Compose([
        # Spatial augmentations
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1, neg=1,
            num_samples=4,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.26, 0.26, 0.26),  # approximately 15 degrees
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),

        # Intensity augmentations (image only)
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5),
        ),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),

        EnsureTyped(keys=["image", "label"]),
    ])


# Usage example
transforms = get_monai_augmentation_pipeline(patch_size=(96, 96, 96))
```

---

## Regulatory and Clinical Considerations

### FDA Clearance Pathways

Medical AI products in the United States require FDA authorization before clinical use. Three main pathways:

**510(k) Clearance:**
- Most common pathway for medical AI.
- Requires demonstration of **substantial equivalence** to a legally marketed predicate device.
- Does not require clinical trials in most cases; bench testing and retrospective validation may suffice.
- Typical timeline: 3-6 months.
- Examples: AI-assisted detection of lung nodules, diabetic retinopathy screening.

**De Novo Classification:**
- For novel devices without a predicate.
- Requires demonstration of safety and effectiveness through more rigorous evidence.
- Creates a new regulatory classification, which becomes a predicate for future 510(k) submissions.
- Typical timeline: 6-12 months.

**PMA (Premarket Approval):**
- Most stringent pathway for Class III (highest risk) devices.
- Requires prospective clinical trials demonstrating safety and effectiveness.
- Typical timeline: 1-3 years.
- Rare for AI/ML-based devices currently.

**Predetermined Change Control Plan (PCCP):**
- FDA framework (since 2023) allowing manufacturers to specify in advance how their AI model will be updated post-market.
- Enables continuous learning without requiring a new submission for each model update.
- Must define the types of changes, validation protocol, and performance guardrails.

### EU MDR and AI Act

**EU Medical Device Regulation (MDR 2017/745):**
- Replaced the Medical Device Directive (MDD) in 2021.
- AI-based medical devices classified as Class IIa or higher, requiring a **Notified Body** assessment.
- Requires clinical evaluation, post-market surveillance, and a Quality Management System (QMS).
- Unique Device Identification (UDI) mandatory.

**EU AI Act:**
- World's first comprehensive AI regulation (entered into force 2024).
- Medical AI is classified as **high-risk**, requiring:
  - Risk management system.
  - Data governance and quality requirements.
  - Technical documentation.
  - Human oversight provisions.
  - Transparency and logging requirements.
  - Accuracy, robustness, and cybersecurity requirements.

### Explainability for Clinical AI

Clinical adoption requires that AI decisions be interpretable to clinicians:

- **GradCAM for 3D**: Compute class-discriminative heatmaps by weighting feature maps by their gradient importance. Extends naturally to 3D volumes.
- **Attention maps**: Extract and visualize attention weights from Attention U-Net or Transformer architectures.
- **Saliency maps**: Compute input gradients to identify which voxels most influence the prediction.
- **SHAP values**: Model-agnostic feature importance for radiomics-based models.
- **Counterfactual explanations**: Show what minimal change to the input would change the prediction.

Practical considerations:

- Clinicians prefer spatial heatmaps overlaid on the original image.
- Explanations must align with clinical knowledge to build trust.
- Regulatory submissions increasingly require explainability documentation.

### Dataset Bias and Fairness

Medical AI systems can encode and amplify biases present in training data:

- **Demographic bias**: Models trained predominantly on one demographic may underperform on others. For example, chest X-ray models trained mostly on adult males may fail on pediatric or female patients.
- **Scanner bias**: Models can overfit to characteristics of specific scanner manufacturers or acquisition protocols.
- **Label bias**: Diagnostic labels reflect the biases of the annotating clinicians and the patient population.
- **Prevalence shift**: A model trained at a high-prevalence center may have poor positive predictive value at a low-prevalence center.

Mitigation strategies:

- Stratified evaluation across demographic groups (age, sex, race, scanner type).
- Multi-site training data with diverse patient populations.
- Domain adaptation and harmonization techniques.
- Fairness-aware training objectives.
- Transparent reporting of model performance across subgroups (Model Cards, Datasheets for Datasets).

**HIPAA Compliance:**

- All patient data must be **de-identified** before use in research (Safe Harbor: remove 18 identifier types; Expert Determination).
- Data Use Agreements (DUA) required for sharing between institutions.
- Protected Health Information (PHI) must be stored in encrypted, access-controlled systems.
- DICOM files contain PHI in metadata headers -- must scrub before sharing.
- Cloud processing requires Business Associate Agreements (BAA) with cloud providers.

---

## See Also

- **CNN Fundamentals** (01): Convolutional layers, pooling, and basic architectures that 3D CNNs build upon.
- **CNN Architectures** (02): ResNet, DenseNet, and SE-Net designs that are extended to 3D.
- **Object Detection and Segmentation** (03): 2D detection and segmentation methods (YOLO, Mask R-CNN, U-Net) that precede their 3D counterparts.
- **Transformers and Attention**: Vision Transformers (ViT), UNETR, and SwinUNETR for medical imaging.
- **Self-Supervised Learning**: Pretraining strategies for medical imaging where labeled data is scarce (contrastive learning, masked image modeling).
- **Generative Models**: Diffusion models for medical image synthesis and augmentation.
- **MLOps and Deployment**: Model serving, monitoring, and updating for clinical AI systems.

---

## Resources

**Key Papers:**
- Cicek et al. (2016) -- "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
- Milletari et al. (2016) -- "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
- Isensee et al. (2021) -- "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
- Qi et al. (2017) -- "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
- Qi et al. (2017) -- "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
- Mildenhall et al. (2020) -- "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
- Kerbl et al. (2023) -- "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
- Mueller et al. (2022) -- "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
- Carreira and Zisserman (2017) -- "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" (I3D)
- Oktay et al. (2018) -- "Attention U-Net: Learning Where to Look for the Pancreas"

**Libraries and Frameworks:**
- MONAI: https://monai.io/ -- Medical imaging deep learning framework
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet -- Self-configuring segmentation
- PyRadiomics: https://pyradiomics.readthedocs.io/ -- Radiomics feature extraction
- pydicom: https://pydicom.github.io/ -- DICOM file handling in Python
- nibabel: https://nipy.org/nibabel/ -- NIfTI and other neuroimaging formats
- SimpleITK: https://simpleitk.org/ -- Medical image processing
- MMDetection3D: https://github.com/open-mmlab/mmdetection3d -- 3D object detection toolbox
- Open3D: https://www.open3d.org/ -- Point cloud and 3D data processing
- ANTs: https://github.com/ANTsX/ANTs -- Advanced image registration
- 3D Slicer: https://www.slicer.org/ -- Medical image visualization and analysis

**Datasets:**
- Medical Segmentation Decathlon: 10 segmentation tasks across organs and modalities
- BraTS: Brain Tumor Segmentation Challenge
- KITTI: Autonomous driving benchmark (LiDAR + camera)
- nuScenes: Large-scale autonomous driving dataset
- Waymo Open Dataset: Autonomous driving with LiDAR and camera
- ModelNet: 3D CAD model classification (40 classes)
- ShapeNet: Large-scale 3D shape repository
- ScanNet: Indoor 3D scene understanding
- LUNA16: Lung nodule detection in CT
- NIH ChestX-ray14: Large-scale chest X-ray classification (2D but often used as medical AI baseline)
- TotalSegmentator: 104 anatomical structures segmented in CT

**Courses and Tutorials:**
- Stanford CS231N: Convolutional Neural Networks for Visual Recognition
- Stanford CS231A: Computer Vision (includes 3D vision)
- MIT 6.S094: Deep Learning for Self-Driving Cars
- MONAI Bootcamp tutorials and documentation
- Medical Image Computing and Computer-Assisted Intervention (MICCAI) tutorials
