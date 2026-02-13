# Computer Vision and Convolutional Neural Networks

**Part IV: Deep Learning for Visual Recognition**

This section covers the fundamentals of computer vision and convolutional neural networks (CNNs), from basic concepts to state-of-the-art architectures and applications.

---

## Contents

### 01. CNN Fundamentals
- Convolutional layers and feature extraction
- Pooling operations
- Padding and stride
- Parameter sharing and spatial hierarchy
- Receptive fields
- PyTorch and TensorFlow implementations

### 02. CNN Architectures
- **Classic Architectures:**
  - LeNet-5 (1998)
  - AlexNet (2012)
  - VGG (2014)
  - GoogLeNet/Inception (2014)

- **Modern Architectures:**
  - ResNet and Skip Connections (2015)
  - DenseNet (2017)
  - MobileNet (2017)
  - EfficientNet (2019)
  - Vision Transformers - ViT (2020)
  - ConvNeXt (2022)

- **2025 State-of-the-Art:**
  - EfficientNetV2
  - Swin Transformer
  - BEiT, DeiT III
  - Hybrid CNN-Transformer models

### 03. Object Detection and Segmentation
- **Object Detection:**
  - R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN)
  - YOLO (You Only Look Once) family - v1 to v10
  - SSD (Single Shot Detector)
  - RetinaNet and Focal Loss
  - DETR (Detection Transformer)

- **Semantic Segmentation:**
  - Fully Convolutional Networks (FCN)
  - U-Net
  - DeepLab and atrous convolution
  - Mask R-CNN
  - Segment Anything Model (SAM 2025)

- **Instance Segmentation:**
  - Mask R-CNN
  - YOLACT
  - SOLOv2

---

## Key Learning Objectives

After completing this section, you will understand:

1. How CNNs extract hierarchical features from images
2. The evolution of CNN architectures and design principles
3. Trade-offs between accuracy, speed, and model size
4. Object detection and segmentation techniques
5. Transfer learning for computer vision tasks
6. 2025 best practices for production computer vision systems

---

## Prerequisites

- Deep Learning Fundamentals (Part III)
- Linear algebra (matrices, convolutions)
- Basic understanding of image formats and preprocessing
- PyTorch or TensorFlow experience

---

## Applications Covered

- **Image Classification**: ImageNet, custom datasets
- **Object Detection**: Real-time detection, autonomous vehicles
- **Semantic Segmentation**: Medical imaging, satellite imagery
- **Instance Segmentation**: Robotics, video analysis
- **Transfer Learning**: Fine-tuning pretrained models
- **Data Augmentation**: Techniques for computer vision

---

## 2025 State-of-the-Art Highlights

- **Vision Transformers** have matched or exceeded CNN performance on many benchmarks
- **Hybrid architectures** (CNN + Transformer) provide best of both worlds
- **Foundation models** like SAM (Segment Anything) enable zero-shot segmentation
- **Efficient models** like EfficientNetV2 and MobileNetV3 enable edge deployment
- **Self-supervised learning** (DINO, MAE) reduces need for labeled data

---

## Quick Reference

| Architecture | Year | Parameters | Top-1 Accuracy | Use Case |
|-------------|------|-----------|----------------|----------|
| AlexNet | 2012 | 61M | 63.3% | Historical |
| VGG-16 | 2014 | 138M | 71.5% | Feature extraction |
| ResNet-50 | 2015 | 25.6M | 76.1% | General purpose |
| EfficientNet-B0 | 2019 | 5.3M | 77.1% | Efficient |
| EfficientNetV2-M | 2021 | 54M | 85.1% | SOTA efficient |
| ViT-B/16 | 2020 | 86M | 84.5% | Transformer |
| Swin-B | 2021 | 88M | 85.2% | Hierarchical ViT |

---

## File Organization

```
04_Computer_Vision_CNNs/
+---- README.md (this file)
+---- 01_CNN_Fundamentals.md
+---- 02_CNN_Architectures.md
+---- 03_Object_Detection_Segmentation.md
```

---

**Last Updated:** 2025-10-14
**Status:** Production-Ready, 2025 State-of-the-Art
