# Object Detection and Segmentation

**Author:** ML Encyclopedia Project
**Last Updated:** 2025
**Prerequisites:** CNN Fundamentals, CNN Architectures
**Difficulty:** PhD-level with Practical Focus

---

## Table of Contents

1. [Introduction](#introduction)
2. [Object Detection Fundamentals](#object-detection-fundamentals)
3. [R-CNN Family](#r-cnn-family)
4. [YOLO Family](#yolo-family)
5. [Single Shot Detectors](#single-shot-detectors)
6. [Detection Transformers](#detection-transformers)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Semantic Segmentation](#semantic-segmentation)
9. [Instance Segmentation](#instance-segmentation)
10. [Panoptic Segmentation](#panoptic-segmentation)
11. [Segment Anything Model (SAM)](#segment-anything-model-sam)
12. [Production Deployment](#production-deployment)
13. [2025 Best Practices](#2025-best-practices)

---

## Introduction

**Object Detection**: Locate and classify multiple objects in an image
- Output: Bounding boxes + class labels
- Example: Detect all cars, pedestrians, traffic signs

**Semantic Segmentation**: Classify each pixel
- Output: Pixel-wise class labels
- No distinction between instances (all cars labeled "car")

**Instance Segmentation**: Detect and segment individual object instances
- Output: Pixel-wise masks + instance IDs
- Distinguishes between different cars

**Panoptic Segmentation**: Unified semantic + instance segmentation
- Both "stuff" (sky, grass) and "things" (cars, people)

**Timeline:**
```
2013: R-CNN (53.3% mAP on PASCAL VOC)
2014: SPPNet, Fast R-CNN
2015: Faster R-CNN (two-stage detector)
2016: YOLO, SSD (one-stage detectors)
2017: RetinaNet (Focal Loss), Mask R-CNN (instance segmentation)
2020: DETR (Detection Transformer)
2021-2023: YOLO v5-v8 (real-time detection)
2023: SAM (Segment Anything)
2024-2025: YOLO v9-v10, SAM 2
```

---

## Object Detection Fundamentals

### Task Definition

**Input**: Image I of size H × W × 3
**Output**: Set of detections {(b_i, c_i, s_i)}

Where:
- `b_i = (x, y, w, h)`: Bounding box coordinates
- `c_i`: Class label (1 to C)
- `s_i`: Confidence score [0, 1]

### Core Challenges

1. **Multiple scales**: Objects at different sizes
2. **Multiple aspect ratios**: Wide (cars) vs. tall (people)
3. **Occlusion**: Partially hidden objects
4. **Class imbalance**: Background >> objects
5. **Real-time inference**: Many applications need 30+ FPS

### Two Paradigms

**Two-stage detectors** (R-CNN family):
1. Generate region proposals
2. Classify each proposal
- Pros: High accuracy
- Cons: Slower inference

**One-stage detectors** (YOLO, SSD, RetinaNet):
1. Single forward pass for detection
- Pros: Fast inference
- Cons: Historically lower accuracy (now competitive)

---

## R-CNN Family

### R-CNN (2014)

**Paper**: "Rich feature hierarchies for accurate object detection" (Girshick et al.)

**Architecture**:
```
Image
→ Selective Search (~2k region proposals)
→ Warp each region to 224×224
→ CNN feature extraction (AlexNet)
→ SVM classifier per class
→ Bounding box regression
```

**Drawbacks**:
- Slow: Separate CNN forward pass for each proposal
- Training pipeline: Multi-stage (CNN, SVM, bbox regression)
- Disk intensive: Features cached to disk

### Fast R-CNN (2015)

**Key Innovation**: Share computation across proposals

**Architecture**:
```
Image
→ CNN (entire image) → Feature map
→ Selective Search proposals
→ RoI Pooling (extract fixed-size features)
→ FC layers
→ Softmax classifier + Bbox regressor
```

**RoI Pooling**: Convert variable-size RoI to fixed-size feature map

```python
import torch
import torch.nn as nn
import torchvision.ops as ops

class RoIPooling(nn.Module):
    """
    Region of Interest Pooling.

    Converts variable-size RoI to fixed-size feature map.
    """
    def __init__(self, output_size=(7, 7), spatial_scale=1/16):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        """
        Args:
            features: [B, C, H, W] feature map
            rois: [N, 5] (batch_idx, x1, y1, x2, y2)

        Returns:
            pooled: [N, C, output_h, output_w]
        """
        return ops.roi_pool(features, rois, self.output_size, self.spatial_scale)
```

**Improvements over R-CNN**:
- 9x faster training
- 140x faster inference
- Higher accuracy
- Single-stage training

### Faster R-CNN (2015)

**Key Innovation**: **Region Proposal Network (RPN)** - learn proposals instead of selective search

**Architecture**:
```
Image
→ Backbone CNN → Feature map
→ RPN (generates proposals)
→ RoI Pooling
→ Classification + Bbox Regression
```

**Region Proposal Network (RPN)**:

```python
class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network (RPN).

    Generates object proposals using anchors.

    Anchors: Predefined boxes at multiple scales and aspect ratios.
    For each anchor: Predict objectness + bbox refinement.
    """
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()

        # 3×3 conv for feature transformation
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Classification: objectness (object vs. background)
        self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1)

        # Regression: bbox refinement (dx, dy, dw, dh)
        self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] feature map

        Returns:
            objectness: [B, num_anchors*2, H, W]
            bbox_deltas: [B, num_anchors*4, H, W]
        """
        x = self.conv(features)
        x = self.relu(x)

        objectness = self.cls_logits(x)
        bbox_deltas = self.bbox_pred(x)

        return objectness, bbox_deltas


def generate_anchors(base_size=16, scales=[8, 16, 32], ratios=[0.5, 1, 2]):
    """
    Generate anchor boxes.

    Args:
        base_size: Base anchor size
        scales: Anchor scales (e.g., 128, 256, 512 pixels)
        ratios: Aspect ratios (width/height)

    Returns:
        anchors: [num_anchors, 4] (x1, y1, x2, y2)
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            h = base_size * scale * torch.sqrt(torch.tensor(ratio))
            w = base_size * scale / torch.sqrt(torch.tensor(ratio))

            x1 = -w / 2
            y1 = -h / 2
            x2 = w / 2
            y2 = h / 2

            anchors.append([x1, y1, x2, y2])

    return torch.tensor(anchors)

# Example: Generate 9 anchors (3 scales × 3 ratios)
anchors = generate_anchors()
print(f"Anchors shape: {anchors.shape}")  # [9, 4]
```

**Complete Faster R-CNN** (Simplified):

```python
class FasterRCNN(nn.Module):
    """
    Faster R-CNN object detector.

    Components:
    1. Backbone: Feature extraction (ResNet, VGG, etc.)
    2. RPN: Proposa generation
    3. RoI Head: Classification and bbox regression
    """
    def __init__(self, backbone, num_classes):
        super().__init__()

        self.backbone = backbone
        self.rpn = RegionProposalNetwork(in_channels=1024, num_anchors=9)

        # RoI pooling
        self.roi_pool = ops.RoIPool(output_size=(7, 7), spatial_scale=1/16)

        # Classification and bbox regression head
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

    def forward(self, images, targets=None):
        # Extract features
        features = self.backbone(images)

        # Generate proposals
        objectness, bbox_deltas = self.rpn(features)

        # Apply proposals to generate RoIs
        # (In practice, use NMS to filter proposals)
        proposals = self._generate_proposals(objectness, bbox_deltas)

        # RoI pooling
        pooled_features = self.roi_pool(features, proposals)

        # Flatten
        pooled_features = pooled_features.flatten(1)

        # Classification head
        x = torch.relu(self.fc1(pooled_features))
        x = torch.relu(self.fc2(x))

        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)

        if self.training:
            # Compute losses
            losses = self._compute_losses(cls_scores, bbox_preds, targets)
            return losses
        else:
            # Post-processing (NMS, etc.)
            detections = self._post_process(cls_scores, bbox_preds, proposals)
            return detections

    def _generate_proposals(self, objectness, bbox_deltas):
        # Simplified: In practice, apply NMS and top-k selection
        # Returns: [N, 5] (batch_idx, x1, y1, x2, y2)
        pass

    def _compute_losses(self, cls_scores, bbox_preds, targets):
        # RPN loss: objectness + bbox regression
        # RoI head loss: classification + bbox regression
        pass

    def _post_process(self, cls_scores, bbox_preds, proposals):
        # Apply NMS per class
        # Return final detections
        pass

# Usage with pretrained model (PyTorch)
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference
images = [torch.rand(3, 800, 800)]
predictions = model(images)

print(predictions[0].keys())  # dict_keys(['boxes', 'labels', 'scores'])
```

**Key Concepts**:

1. **Anchors**: Predefined boxes at multiple scales/ratios
2. **RPN**: Learns to propose regions
3. **RoI Pooling**: Fixed-size features from variable-size regions
4. **Multi-task loss**: Classification + bbox regression

**Performance**:
- PASCAL VOC 2007: 73.2% mAP
- Inference: ~5 FPS (slower than one-stage)

---

## YOLO Family

**YOLO**: "You Only Look Once" - single-pass object detection

**Key Idea**: Frame detection as regression problem
- Divide image into grid
- Each grid cell predicts bounding boxes + class probabilities
- Single CNN forward pass

### YOLO v1 (2016)

**Architecture**:
```
Image (448×448)
→ CNN (24 conv layers + 2 FC)
→ Reshape to 7×7×30
→ Each cell: 2 boxes × (5 coords + conf) + 20 class probs
```

**Loss Function**:
```
L = λ_coord Σ (x,y errors)
  + λ_coord Σ (w,h errors)
  + Σ (confidence errors for boxes with objects)
  + λ_noobj Σ (confidence errors for boxes without objects)
  + Σ (class probability errors)
```

**Limitations**:
- Struggles with small objects
- Limited by grid size (7×7 = max 49 objects)
- Low recall compared to Faster R-CNN

### YOLO v2 / YOLO9000 (2016)

**Improvements**:
- Batch normalization
- High-resolution classifier (448×448)
- Anchor boxes (from Faster R-CNN)
- Dimension clusters (k-means on training boxes)
- Multi-scale training
- Passthrough layer (like skip connections)

**Results**: 78.6% mAP on VOC 2007 (vs. 73.2% for Faster R-CNN)

### YOLO v3 (2018)

**Major improvements**:
- **Multi-scale predictions**: 3 different scales (like FPN)
- **Darknet-53 backbone**: 53-layer network
- **Binary cross-entropy**: Instead of softmax (allows multi-label)

```python
class YOLOv3(nn.Module):
    """
    YOLOv3 architecture (simplified).

    Multi-scale detection at 3 different scales:
    - Small objects: 52×52 grid
    - Medium objects: 26×26 grid
    - Large objects: 13×13 grid
    """
    def __init__(self, num_classes=80):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = 3  # Per scale

        # Darknet-53 backbone (simplified)
        self.backbone = self._make_darknet53()

        # Detection heads at 3 scales
        self.head_large = self._make_detection_head(1024, num_classes)   # 13×13
        self.head_medium = self._make_detection_head(512, num_classes)   # 26×26
        self.head_small = self._make_detection_head(256, num_classes)    # 52×52

    def _make_darknet53(self):
        # Simplified: Use ResNet-like structure
        return nn.Sequential(
            # Conv layers with residual connections
            # Returns multi-scale features
        )

    def _make_detection_head(self, in_channels, num_classes):
        """
        Detection head: Predicts boxes + objectness + classes.

        Output: [B, num_anchors * (5 + num_classes), H, W]
        - 5: (x, y, w, h, objectness)
        - num_classes: class probabilities
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels * 2, self.num_anchors * (5 + num_classes), 1)
        )

    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        feat_small, feat_medium, feat_large = features

        # Predictions at each scale
        pred_large = self.head_large(feat_large)    # 13×13
        pred_medium = self.head_medium(feat_medium)  # 26×26
        pred_small = self.head_small(feat_small)    # 52×52

        return pred_large, pred_medium, pred_small

# YOLO v3 loss function
class YOLOv3Loss(nn.Module):
    """
    YOLOv3 loss function.

    Components:
    1. Localization loss (bbox coordinates)
    2. Objectness loss (confidence)
    3. Classification loss (class probabilities)
    """
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output
            targets: Ground truth boxes + labels

        Returns:
            total_loss: Sum of all loss components
        """
        # Parse predictions
        pred_boxes, pred_obj, pred_cls = self._parse_predictions(predictions)

        # Match predictions to targets
        matched_pred, matched_target = self._match_predictions(pred_boxes, targets)

        # Compute losses
        loc_loss = self._localization_loss(matched_pred, matched_target)
        obj_loss = self._objectness_loss(pred_obj, matched_target)
        cls_loss = self._classification_loss(pred_cls, matched_target)

        # Total loss
        total_loss = loc_loss + obj_loss + cls_loss

        return total_loss

    def _localization_loss(self, pred_boxes, target_boxes):
        """
        Bounding box regression loss.

        Uses MSE for (x, y) and (w, h).
        """
        return self.mse_loss(pred_boxes, target_boxes)

    def _objectness_loss(self, pred_obj, targets):
        """
        Objectness (confidence) loss.

        Binary cross-entropy.
        """
        return self.bce_loss(pred_obj, targets)

    def _classification_loss(self, pred_cls, targets):
        """
        Classification loss.

        Binary cross-entropy (multi-label).
        """
        return self.bce_loss(pred_cls, targets)
```

### YOLO v4 (2020)

**Contributions** (Bag of Freebies + Bag of Specials):

**Bag of Freebies** (no inference cost):
- Mosaic data augmentation
- Self-adversarial training
- CIoU loss
- Cross mini-batch normalization

**Bag of Specials** (small inference cost):
- Mish activation
- CSPDarknet53 backbone
- SPP (Spatial Pyramid Pooling)
- PANet (Path Aggregation Network)

**Results**: 43.5% AP on COCO (vs. 33% for YOLOv3)

### YOLO v5-v8 (2020-2023)

**YOLOv5** (Ultralytics):
- PyTorch implementation (v1-v4 were Darknet)
- Easy to use, well-documented
- Multiple variants (nano, small, medium, large, xlarge)

**YOLOv7** (2022):
- Trainable bag-of-freebies
- E-ELAN (Extended Efficient Layer Aggregation Network)
- Model scaling for concatenation-based models

**YOLOv8** (2023):
- Anchor-free detection
- Decoupled head (separate classification/localization)
- New backbone and neck
- Improved loss function

**Production-Ready PyTorch Implementation** (Using Ultralytics):

```python
from ultralytics import YOLO
import torch
import cv2

# Load pretrained YOLOv8
model = YOLO('yolov8n.pt')  # nano (fastest)
# model = YOLO('yolov8s.pt')  # small
# model = YOLO('yolov8m.pt')  # medium
# model = YOLO('yolov8l.pt')  # large
# model = YOLO('yolov8x.pt')  # xlarge (most accurate)

# Inference
image = cv2.imread('image.jpg')
results = model(image)

# Process results
for result in results:
    boxes = result.boxes  # Boxes object
    for box in boxes:
        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Confidence
        conf = box.conf[0].item()

        # Class
        cls = int(box.cls[0].item())

        print(f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
              f"Conf: {conf:.2f}, Class: {cls}")

# Training on custom dataset
model.train(
    data='custom_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)

# Validation
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Export to different formats
model.export(format='onnx')  # ONNX
model.export(format='torchscript')  # TorchScript
model.export(format='engine')  # TensorRT for production
```

### YOLO v9-v10 (2024-2025)

**YOLOv9** (2024):
- Programmable Gradient Information (PGI)
- Generalized ELAN (GELAN)
- Better gradient flow

**YOLOv10** (2025):
- NMS-free detection (end-to-end)
- Spatial-channel decoupled downsampling
- Rank-guided block design
- **SOTA real-time detection**

**Performance Comparison** (COCO val):

| Model | Size | mAP50-95 | FPS (V100) | Params |
|-------|------|----------|------------|--------|
| YOLOv3 | 640 | 43.3% | 45 | 62M |
| YOLOv5x | 640 | 50.7% | 50 | 86M |
| YOLOv8x | 640 | 53.9% | 50 | 68M |
| YOLOv9-C | 640 | 53.0% | 65 | 51M |
| YOLOv10-X | 640 | 54.4% | 70 | 32M |

**2025 Recommendation**: **YOLOv10** for real-time detection

---

## Single Shot Detectors

### SSD (2016)

**Paper**: "SSD: Single Shot MultiBox Detector" (Liu et al.)

**Key Ideas**:
- Multi-scale feature maps for detection
- Default boxes (anchors) at multiple scales/ratios
- Single forward pass

**Architecture**:
```
Image → VGG-16 (base network)
        ↓
        Conv4_3 → Detection (38×38)
        Conv7 → Detection (19×19)
        Conv8_2 → Detection (10×10)
        Conv9_2 → Detection (5×5)
        Conv10_2 → Detection (3×3)
        Conv11_2 → Detection (1×1)
```

**Advantages**:
- Fast (59 FPS on 300×300 images)
- Good accuracy (74.3% mAP on VOC 2007)

**Disadvantages**:
- Struggles with small objects
- Hard negative mining required

### RetinaNet (2017)

**Paper**: "Focal Loss for Dense Object Detection" (Lin et al.)

**Key Innovation**: **Focal Loss** - addresses class imbalance

**Problem**: One-stage detectors have massive class imbalance
- Background boxes: ~100k
- Foreground boxes: ~100
- Easy negatives dominate loss

**Solution**: Focal Loss
```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

Where:
- p_t: predicted probability for true class
- α_t: balancing factor (0.25 typical)
- γ: focusing parameter (2 typical)
```

**Effect**: Down-weight easy examples, focus on hard negatives

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Args:
        alpha: Balancing factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] predicted logits
            targets: [N] ground truth classes (0 to C-1)

        Returns:
            loss: Focal loss
        """
        # Compute softmax probabilities
        p = torch.softmax(inputs, dim=1)

        # Get probability for true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal loss
        focal_weight = (1 - p_t) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(p_t)

        return loss.mean()


class RetinaNet(nn.Module):
    """
    RetinaNet object detector.

    Components:
    1. Backbone: ResNet + FPN (Feature Pyramid Network)
    2. Classification subnet: Predicts class probabilities
    3. Regression subnet: Predicts bbox offsets
    """
    def __init__(self, num_classes=80):
        super().__init__()

        # Backbone: ResNet-50 + FPN
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        # Classification subnet (4 × 3×3 conv + 1 × 3×3 conv)
        self.cls_subnet = self._make_head(256, num_classes * 9)  # 9 anchors per location

        # Regression subnet (4 × 3×3 conv + 1 × 3×3 conv)
        self.box_subnet = self._make_head(256, 4 * 9)  # 4 coords × 9 anchors

        # Focal loss
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def _make_head(self, in_channels, out_channels):
        """
        Prediction head: 4 conv layers with 256 filters + final conv.
        """
        layers = []
        for _ in range(4):
            layers.extend([
                nn.Conv2d(in_channels, 256, 3, padding=1),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Conv2d(256, out_channels, 3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, images, targets=None):
        # Extract multi-scale features
        features = self.backbone(images)

        # Predictions at each level
        cls_preds = []
        box_preds = []

        for feat in features.values():
            cls_preds.append(self.cls_subnet(feat))
            box_preds.append(self.box_subnet(feat))

        if self.training:
            # Compute losses
            losses = self._compute_losses(cls_preds, box_preds, targets)
            return losses
        else:
            # Post-processing
            detections = self._post_process(cls_preds, box_preds)
            return detections

    def _compute_losses(self, cls_preds, box_preds, targets):
        # Focal loss for classification
        # Smooth L1 loss for bbox regression
        pass

# Usage with pretrained model
from torchvision.models.detection import retinanet_resnet50_fpn

model = retinanet_resnet50_fpn(pretrained=True)
model.eval()

images = [torch.rand(3, 800, 800)]
predictions = model(images)
```

**Results**:
- COCO test-dev: 39.1% AP
- Speed: ~5 FPS (slower than YOLO but more accurate)
- **Key contribution**: Focal Loss now used in many detectors

---

## Detection Transformers

### DETR (2020)

**Paper**: "End-to-End Object Detection with Transformers" (Carion et al., Facebook AI)

**Key Innovation**: **End-to-end detection** without NMS or anchors

**Architecture**:
```
Image
→ CNN Backbone (ResNet) → Feature map
→ Flatten + Positional Encoding
→ Transformer Encoder
→ Transformer Decoder (with learnable object queries)
→ Set Prediction (class + bbox)
```

**Object Queries**: Learnable embeddings (e.g., 100 queries for 100 objects)

**Set Prediction**: Bipartite matching between predictions and ground truth

```python
class DETR(nn.Module):
    """
    DETR: DEtection TRansformer.

    End-to-end object detection using Transformers.
    """
    def __init__(self, num_classes=91, num_queries=100, hidden_dim=256):
        super().__init__()

        # Backbone
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Reduce channels
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        # Object queries (learnable)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # (x_center, y_center, width, height)
        )

        # Positional encoding
        self.row_embed = nn.Embedding(50, hidden_dim // 2)
        self.col_embed = nn.Embedding(50, hidden_dim // 2)

    def forward(self, images):
        # Extract features
        features = self.backbone(images)  # [B, 2048, H, W]
        features = self.conv(features)    # [B, 256, H, W]

        # Positional encoding
        h, w = features.shape[-2:]
        pos = self._get_positional_encoding(h, w)
        pos = pos.unsqueeze(0).repeat(features.shape[0], 1, 1, 1)

        # Flatten spatial dimensions
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, H*W, 256]
        pos_flat = pos.flatten(2).permute(0, 2, 1)

        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(features.shape[0], 1, 1)

        # Transformer
        memory = self.transformer.encoder(features_flat + pos_flat)
        hs = self.transformer.decoder(query_embed, memory)

        # Predictions
        outputs_class = self.class_embed(hs)  # [B, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [B, num_queries, 4]

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

    def _get_positional_encoding(self, h, w):
        """2D positional encoding."""
        row_pos = self.row_embed(torch.arange(h, device=self.row_embed.weight.device))
        col_pos = self.col_embed(torch.arange(w, device=self.col_embed.weight.device))

        # Combine row and column embeddings
        pos = torch.cat([
            row_pos.unsqueeze(1).repeat(1, w, 1),
            col_pos.unsqueeze(0).repeat(h, 1, 1)
        ], dim=-1)

        return pos.permute(2, 0, 1)  # [256, H, W]


class HungarianMatcher(nn.Module):
    """
    Hungarian matching for bipartite matching between predictions and targets.

    Computes optimal assignment that minimizes matching cost.
    """
    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dict with 'pred_logits' [B, num_queries, num_classes]
                     and 'pred_boxes' [B, num_queries, 4]
            targets: List of dicts with 'labels' and 'boxes'

        Returns:
            indices: List of (pred_idx, target_idx) tuples
        """
        from scipy.optimize import linear_sum_assignment

        bs, num_queries = outputs['pred_logits'].shape[:2]

        # Flatten batch dimension
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)

        # Concatenate targets
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        # Classification cost
        cost_class = -out_prob[:, tgt_ids]

        # L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -self._generalized_box_iou(out_bbox, tgt_bbox)

        # Total cost
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Hungarian matching
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split([len(v['boxes']) for v in targets], -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]

    def _generalized_box_iou(self, boxes1, boxes2):
        """Compute GIoU between boxes."""
        # Implementation of Generalized IoU
        pass

# Usage
model = DETR(num_classes=91, num_queries=100)
images = torch.rand(2, 3, 800, 800)
outputs = model(images)

print(f"Class predictions: {outputs['pred_logits'].shape}")  # [2, 100, 92]
print(f"Box predictions: {outputs['pred_boxes'].shape}")     # [2, 100, 4]
```

**Advantages**:
- No hand-designed components (anchors, NMS)
- Directly outputs set of predictions
- Strong performance on large objects

**Disadvantages**:
- Slow convergence (500 epochs)
- Lower performance on small objects
- Higher computational cost

**Variants**:
- **Deformable DETR** (2020): Faster convergence, better small object detection
- **DINO** (2022): Improved training, better accuracy
- **Grounding DINO** (2023): Open-vocabulary detection

---

## Evaluation Metrics

### Intersection over Union (IoU)

```python
def compute_iou(box1, box2):
    """
    Compute Intersection over Union.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        iou: Intersection over Union
    """
    # Intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def generalized_iou(box1, box2):
    """
    Generalized IoU (GIoU).

    GIoU = IoU - |C \ (A ∪ B)| / |C|

    Where C is smallest enclosing box.
    """
    iou = compute_iou(box1, box2)

    # Smallest enclosing box
    x1_c = min(box1[0], box2[0])
    y1_c = min(box1[1], box2[1])
    x2_c = max(box1[2], box2[2])
    y2_c = max(box1[3], box2[3])

    c_area = (x2_c - x1_c) * (y2_c - y1_c)

    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    inter_area = iou * (box1_area + box2_area - iou * (box1_area + box2_area))
    union_area = box1_area + box2_area - inter_area

    giou = iou - (c_area - union_area) / c_area
    return giou
```

### Mean Average Precision (mAP)

**Precision**: TP / (TP + FP)
**Recall**: TP / (TP + FN)

**Average Precision (AP)**: Area under Precision-Recall curve

**Mean Average Precision (mAP)**: Average AP across all classes

**COCO metrics**:
- **AP / mAP**: Average over IoU thresholds [0.5:0.95:0.05]
- **AP50**: AP at IoU=0.5
- **AP75**: AP at IoU=0.75
- **AP_S**: AP for small objects (area < 32²)
- **AP_M**: AP for medium objects (32² < area < 96²)
- **AP_L**: AP for large objects (area > 96²)

```python
def compute_ap(precisions, recalls):
    """
    Compute Average Precision (AP).

    Args:
        precisions: List of precision values
        recalls: List of recall values

    Returns:
        ap: Average precision
    """
    # Sort by recall
    sorted_indices = sorted(range(len(recalls)), key=lambda i: recalls[i])
    sorted_precisions = [precisions[i] for i in sorted_indices]
    sorted_recalls = [recalls[i] for i in sorted_indices]

    # Compute AP (area under curve)
    ap = 0.0
    for i in range(1, len(sorted_recalls)):
        delta_recall = sorted_recalls[i] - sorted_recalls[i-1]
        ap += sorted_precisions[i] * delta_recall

    return ap


def compute_map(detections, ground_truths, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP).

    Args:
        detections: List of predicted boxes [class, score, x1, y1, x2, y2]
        ground_truths: List of ground truth boxes [class, x1, y1, x2, y2]
        iou_threshold: IoU threshold for matching

    Returns:
        map: Mean Average Precision
    """
    # Compute AP for each class
    aps = []
    for class_id in set([gt[0] for gt in ground_truths]):
        # Filter detections and ground truths for this class
        class_dets = [d for d in detections if d[0] == class_id]
        class_gts = [gt for gt in ground_truths if gt[0] == class_id]

        # Sort detections by confidence
        class_dets.sort(key=lambda x: x[1], reverse=True)

        # Match detections to ground truths
        tp = []
        fp = []
        matched_gts = set()

        for det in class_dets:
            det_box = det[2:]
            max_iou = 0
            max_gt_idx = -1

            for gt_idx, gt in enumerate(class_gts):
                if gt_idx in matched_gts:
                    continue

                gt_box = gt[1:]
                iou = compute_iou(det_box, gt_box)

                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            if max_iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
                matched_gts.add(max_gt_idx)
            else:
                tp.append(0)
                fp.append(1)

        # Compute precision and recall
        cumsum_tp = torch.cumsum(torch.tensor(tp), dim=0)
        cumsum_fp = torch.cumsum(torch.tensor(fp), dim=0)

        precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)
        recalls = cumsum_tp / len(class_gts)

        # Compute AP
        ap = compute_ap(precisions.tolist(), recalls.tolist())
        aps.append(ap)

    # Mean AP
    return sum(aps) / len(aps) if aps else 0.0
```

### Non-Maximum Suppression (NMS)

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression.

    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        scores: [N] confidence scores
        iou_threshold: IoU threshold

    Returns:
        keep: Indices of boxes to keep
    """
    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        # Keep highest scoring box
        idx = sorted_indices[0]
        keep.append(idx.item())

        if len(sorted_indices) == 1:
            break

        # Compute IoU with remaining boxes
        box = boxes[idx]
        remaining_boxes = boxes[sorted_indices[1:]]

        ious = torch.tensor([compute_iou(box, rb) for rb in remaining_boxes])

        # Keep boxes with IoU < threshold
        keep_mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]

    return keep

# Efficient PyTorch implementation
from torchvision.ops import nms as torch_nms

# Usage
boxes = torch.tensor([[10, 10, 50, 50], [15, 15, 55, 55], [100, 100, 150, 150]])
scores = torch.tensor([0.9, 0.8, 0.95])

keep = torch_nms(boxes, scores, iou_threshold=0.5)
print(keep)  # tensor([2, 0])
```

---

## Semantic Segmentation

**Task**: Classify each pixel into semantic categories

**Output**: Segmentation map of same size as input, each pixel labeled with class

### Fully Convolutional Networks (FCN) (2015)

**Key Innovation**: Replace FC layers with convolutional layers

**Architecture**:
```
Image
→ CNN Encoder (VGG-16)
→ 1×1 Convolutions (replace FC layers)
→ Upsampling (transposed convolutions)
→ Skip connections from encoder
→ Pixel-wise prediction
```

```python
class FCN(nn.Module):
    """
    Fully Convolutional Network for semantic segmentation.
    """
    def __init__(self, num_classes=21):
        super().__init__()

        # Encoder (VGG-16 backbone)
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True)
        features = list(vgg.features.children())

        self.enc1 = nn.Sequential(*features[:5])   # Pool1
        self.enc2 = nn.Sequential(*features[5:10])  # Pool2
        self.enc3 = nn.Sequential(*features[10:17]) # Pool3
        self.enc4 = nn.Sequential(*features[17:24]) # Pool4
        self.enc5 = nn.Sequential(*features[24:])   # Pool5

        # 1×1 convolutions (replace FC)
        self.conv6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.conv7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Scoring layer
        self.score = nn.Conv2d(4096, num_classes, 1)

        # Upsampling
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)

        # Score pool layers (for skip connections)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # FC replacement
        x = self.conv6(enc5)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        # Score
        score = self.score(x)

        # Upsample and add skip connections
        upscore2 = self.upscore2(score)
        score_pool4 = self.score_pool4(enc4)
        fuse1 = upscore2 + score_pool4

        upscore8 = self.upscore8(fuse1)

        return upscore8
```

### U-Net (2015)

**Key Innovation**: Symmetric encoder-decoder with skip connections

**Architecture**: U-shaped (hence the name)

```python
class UNet(nn.Module):
    """
    U-Net for semantic segmentation.

    Widely used in medical imaging and general segmentation.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        # Bottleneck
        self.bottleneck = self._block(512, 1024)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._block(1024, 512)  # 512 + 512 (skip connection)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._block(512, 256)  # 256 + 256

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._block(256, 128)  # 128 + 128

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)   # 64 + 64

        # Final output
        self.out = nn.Conv2d(64, num_classes, 1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def _block(self, in_channels, out_channels):
        """Convolutional block: Conv → ReLU → Conv → ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        return self.out(dec1)

# Usage
model = UNet(in_channels=3, num_classes=21)
x = torch.rand(1, 3, 256, 256)
output = model(x)
print(f"Output shape: {output.shape}")  # [1, 21, 256, 256]
```

### DeepLab v3+ (2018)

**Key Innovations**:
1. **Atrous (Dilated) Convolutions**: Expand receptive field without losing resolution
2. **Atrous Spatial Pyramid Pooling (ASPP)**: Multi-scale features
3. **Encoder-Decoder**: With atrous separable convolutions

```python
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Parallel atrous convolutions with different rates.
    """
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        # 1×1 conv
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 3×3 atrous conv, rate=6
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 3×3 atrous conv, rate=12
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 3×3 atrous conv, rate=18
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Fuse all branches
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[2:]

        # Apply all branches
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_1(x)
        feat3 = self.conv3x3_2(x)
        feat4 = self.conv3x3_3(x)
        feat5 = self.global_avg_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=True)

        # Concatenate
        x = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)

        # Fuse
        x = self.conv_fuse(x)

        return x

# Full DeepLab v3+ implementation
# Use torchvision.models.segmentation.deeplabv3_resnet50()
```

**2025 Status**: DeepLab v3+ remains competitive for semantic segmentation

---

## Instance Segmentation

**Task**: Detect and segment individual object instances

**Difference from Semantic Segmentation**:
- Semantic: All "car" pixels labeled as "car"
- Instance: "car_1", "car_2", "car_3" (separate instances)

### Mask R-CNN (2017)

**Paper**: "Mask R-CNN" (He et al.)

**Key Innovation**: Extend Faster R-CNN with mask prediction branch

**Architecture**:
```
Faster R-CNN
├─ Classification branch (class scores)
├─ Bounding box regression branch (bbox coords)
└─ Mask prediction branch (binary mask) ← NEW
```

```python
# Production-ready usage (torchvision)
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch
import cv2
import numpy as np

# Load pretrained Mask R-CNN
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
image_tensor = image_tensor.unsqueeze(0)

with torch.no_grad():
    predictions = model(image_tensor)[0]

# Extract results
boxes = predictions['boxes'].cpu().numpy()
labels = predictions['labels'].cpu().numpy()
scores = predictions['scores'].cpu().numpy()
masks = predictions['masks'].cpu().numpy()

# Visualize
threshold = 0.5
for i, score in enumerate(scores):
    if score > threshold:
        # Bounding box
        box = boxes[i].astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Mask
        mask = masks[i, 0] > 0.5
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = [0, 0, 255]  # Red mask
        image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

cv2.imwrite('result.jpg', image)
```

**Key Components**:

1. **RoIAlign**: Improved version of RoI Pooling (no quantization)
2. **Mask Branch**: Predicts binary mask for each class
3. **Multi-task Loss**: Classification + Bbox + Mask

### YOLACT (2019)

**Paper**: "YOLACT: Real-time Instance Segmentation"

**Key Innovation**: Fast instance segmentation (33 FPS)

**Approach**:
- Generate prototype masks (full image resolution)
- Predict mask coefficients per instance
- Linear combination: mask = Σ coefficients × prototypes

### SOLOv2 (2020)

**Paper**: "SOLOv2: Dynamic and Fast Instance Segmentation"

**Key Innovation**: Segment Objects by Locations

**Approach**:
- Divide image into grid
- Each cell predicts: object category + instance mask
- No bounding boxes or anchor boxes

---

## Panoptic Segmentation

**Task**: Unified semantic + instance segmentation

**Output**:
- "Stuff" classes (sky, grass): Semantic segmentation
- "Thing" classes (cars, people): Instance segmentation

### Panoptic FPN (2019)

**Architecture**: Combine semantic and instance segmentation heads

```python
# Production usage (Detectron2)
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor
#
# cfg = get_cfg()
# cfg.merge_from_file("configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
# cfg.MODEL.WEIGHTS = "model_final.pth"
# predictor = DefaultPredictor(cfg)
#
# outputs = predictor(image)
# panoptic_seg = outputs['panoptic_seg']
```

---

## Segment Anything Model (SAM)

**Paper**: "Segment Anything" (Kirillov et al., Meta AI, 2023)

**Key Innovation**: **Foundation model for segmentation** - segment any object with prompts

**Prompts**:
- Points (positive/negative)
- Bounding boxes
- Text (SAM 2)
- Rough masks

**Architecture**:
```
Image Encoder (ViT-H)
→ Prompt Encoder (points, boxes, masks)
→ Mask Decoder (lightweight)
→ Multiple mask predictions (ambiguous cases)
```

**Production Usage**:

```python
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device='cuda')

predictor = SamPredictor(sam)

# Load image
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set image
predictor.set_image(image_rgb)

# Prompt: Point (x, y)
input_point = np.array([[500, 375]])
input_label = np.array([1])  # 1 = foreground, 0 = background

# Predict masks
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # Get 3 masks with different qualities
)

# Use mask with highest score
best_mask = masks[np.argmax(scores)]

# Visualize
colored_mask = np.zeros_like(image)
colored_mask[best_mask] = [0, 255, 0]
result = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

cv2.imwrite('sam_result.jpg', result)

# Prompt: Bounding box
input_box = np.array([100, 100, 500, 500])  # [x1, y1, x2, y2]
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False
)
```

**SAM 2 (2024-2025)**:
- **Video segmentation**: Temporal consistency
- **Text prompts**: "segment the cat"
- **Better performance**: Improved mask quality
- **Faster inference**: Optimized architecture

**2025 Applications**:
- **Interactive annotation**: Click to segment
- **Data labeling**: Accelerate dataset creation
- **Photo editing**: Automatic object selection
- **Medical imaging**: Segment organs, tumors

---

## Production Deployment

### Model Optimization

```python
# 1. Export to ONNX
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 2. TensorRT for NVIDIA GPUs
# Use trtexec or torch2trt for conversion

# 3. Quantization (INT8)
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# 4. Model pruning
import torch.nn.utils.prune as prune

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

### Inference Pipeline

```python
class ObjectDetectionPipeline:
    """
    Production-ready object detection pipeline.
    """
    def __init__(self, model_path, device='cuda', conf_threshold=0.5):
        self.device = device
        self.conf_threshold = conf_threshold

        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)

    def preprocess(self, image):
        """Preprocess image for inference."""
        # Resize, normalize, etc.
        return image

    def postprocess(self, predictions):
        """Post-process predictions."""
        # NMS, threshold filtering, etc.
        return predictions

    @torch.no_grad()
    def predict(self, image):
        """Run inference."""
        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        predictions = self.model(input_tensor)

        # Post-process
        results = self.postprocess(predictions)

        return results

    def predict_batch(self, images):
        """Batch inference for efficiency."""
        input_tensors = [self.preprocess(img) for img in images]
        batch = torch.stack(input_tensors)

        predictions = self.model(batch)

        results = [self.postprocess(pred) for pred in predictions]
        return results

# Usage
pipeline = ObjectDetectionPipeline('yolov8n.pt', device='cuda')

# Single image
result = pipeline.predict(image)

# Batch (more efficient)
results = pipeline.predict_batch(images)
```

### Real-Time Video Processing

```python
import cv2
from ultralytics import YOLO

def process_video(video_path, output_path, model_path='yolov8n.pt'):
    """
    Process video with object detection.
    """
    # Load model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Write frame
        out.write(annotated_frame)

        # Display (optional)
        cv2.imshow('Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage
process_video('input.mp4', 'output.mp4', 'yolov8n.pt')
```

---

## 2025 Best Practices

### Model Selection

**Real-time detection (30+ FPS)**:
- YOLOv10-N/S for edge devices
- YOLOv10-M/L for GPUs
- TensorRT optimization

**High accuracy**:
- DINO (Transformer-based)
- Mask R-CNN for instance segmentation
- SAM for interactive segmentation

**Production deployment**:
1. **Start with pretrained models**: Always use transfer learning
2. **Fine-tune on custom data**: Even with small datasets
3. **Optimize for inference**: ONNX, TensorRT, quantization
4. **Monitor performance**: Latency, throughput, accuracy
5. **A/B test**: Compare models in production

### Data Augmentation (2025)

```python
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,
    ShiftScaleRotate, CLAHE, Blur, GaussNoise, Normalize
)
from albumentations.pytorch import ToTensorV2

train_transform = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.3),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    CLAHE(p=0.3),
    Blur(blur_limit=3, p=0.3),
    GaussNoise(p=0.3),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
```

### Training Tips

1. **Multi-scale training**: Vary input sizes
2. **Mosaic augmentation**: Combine 4 images (YOLO)
3. **CopyPaste**: Copy objects between images
4. **Auto-augmentation**: Learn augmentation policies
5. **Progressive training**: Start small, increase size
6. **EMA (Exponential Moving Average)**: Stabilize training

### Evaluation Checklist

- [ ] mAP on validation set
- [ ] Per-class AP (identify weak classes)
- [ ] Small/medium/large object performance
- [ ] Inference speed (FPS)
- [ ] Model size (MB)
- [ ] Qualitative results (visualize failures)
- [ ] Edge cases (occlusion, extreme aspect ratios)

---

## Summary

This comprehensive guide covered object detection and segmentation:

**Object Detection**:
1. **R-CNN Family**: Two-stage detectors (high accuracy)
2. **YOLO Family**: One-stage detectors (real-time)
3. **SSD, RetinaNet**: Single-shot with innovations (Focal Loss)
4. **DETR**: End-to-end with Transformers

**Semantic Segmentation**:
5. **FCN**: Fully convolutional networks
6. **U-Net**: Encoder-decoder with skip connections
7. **DeepLab**: Atrous convolutions, ASPP

**Instance Segmentation**:
8. **Mask R-CNN**: Extend Faster R-CNN with masks
9. **YOLACT, SOLOv2**: Real-time instance segmentation

**Foundation Models**:
10. **SAM**: Segment anything with prompts

**2025 Recommendations**:
- **Detection**: YOLOv10 for real-time, DINO for accuracy
- **Segmentation**: SAM for flexibility, Mask R-CNN for instances
- **Production**: ONNX/TensorRT optimization, batch inference

**Key Metrics**: IoU, mAP, AP50, AP75, FPS

**Production**: Model optimization, efficient inference, monitoring

---

**References:**

- Girshick et al. (2014). "Rich feature hierarchies for accurate object detection"
- Ren et al. (2015). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
- Redmon et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection"
- Liu et al. (2016). "SSD: Single Shot MultiBox Detector"
- He et al. (2017). "Mask R-CNN"
- Lin et al. (2017). "Focal Loss for Dense Object Detection"
- Carion et al. (2020). "End-to-End Object Detection with Transformers"
- Kirillov et al. (2023). "Segment Anything"
- Wang et al. (2024). "YOLOv10: Real-Time End-to-End Object Detection"

---

*End of Object Detection and Segmentation*
