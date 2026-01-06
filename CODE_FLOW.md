# DINO Teacher: Code Architecture and Image Flow Documentation

## Overview

**DINO Teacher** is a domain adaptive object detection framework that leverages Vision Foundation Models (VFMs) as a source of pseudo-labels and for cross-domain alignment. The system is built on top of Detectron2 and Adaptive Teacher, using DINO (self-supervised Vision Transformer) models to improve object detection in target domains.

## Repository Structure

```
DINO_Teacher/
├── train_net.py              # Main entry point for training and evaluation
├── configs/                  # Configuration files for different training scenarios
│   ├── vit_labeller.yaml    # Config for training DINO ViT labeller
│   ├── vgg_city2bdd.yaml    # Config for student model training (VGG backbone)
│   └── Base-RCNN-C4.yaml    # Base R-CNN configuration
├── dinoteacher/              # Main DINO Teacher implementation
│   ├── engine/              # Training and inference engines
│   │   ├── trainer.py       # DINOTeacherTrainer - main training loop
│   │   ├── gen_labels.py    # Pseudo-label generation
│   │   ├── build_dino.py    # DINO ViT feature extractor
│   │   └── align_head.py    # Teacher-student alignment module
│   ├── modeling/            # Model architectures
│   │   ├── meta_arch/       # Meta-architectures (RCNN, ViT)
│   │   └── roi_heads/       # ROI head implementations
│   └── data/                # Data loading and augmentation
│       ├── dataset_mapper.py # Custom dataset mapping
│       └── datasets/        # Dataset registration
├── adapteacher/             # Adaptive Teacher base implementation
├── dinov1/                  # DINO v1 model support
├── dinov2/                  # DINO v2 model support
└── weights/                 # Pre-trained model weights (to be downloaded)
```

## High-Level Architecture

The DINO Teacher framework consists of three main phases:

### Phase 1: DINO Labeller Training
- Train a DINO ViT-based detector on source domain (Cityscapes)
- Use Vision Transformer (ViT-L or ViT-G) as backbone
- Generates high-quality pseudo-labels for target domain

### Phase 2: Pseudo-Label Generation
- Use trained DINO labeller to generate pseudo-labels on target domain
- Pseudo-labels saved as pickle files for later use

### Phase 3: Student Model Training
- Train student detector (e.g., VGG or ResNet backbone)
- Use source domain ground truth + target domain pseudo-labels
- Apply cross-domain feature alignment with DINO teacher

## Complete Image Flow: From Input to Output

### 1. Data Loading and Preprocessing

**Entry Point**: `train_net.py` → `DINOTeacherTrainer.build_train_loader()`

```
Input Image File (e.g., /path/to/image.jpg)
    ↓
DatasetMapperTwoCropSeparateKeepTf.__call__()
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. Read Image (RGB format)                                   │
│    - utils.read_image(dataset_dict["file_name"])            │
│                                                               │
│ 2. Apply Weak Augmentation                                   │
│    - Resize to multiple of patch size (for ViT)             │
│    - Random horizontal flip                                  │
│    - Random crop and pad (if enabled)                       │
│                                                               │
│ 3. Transform Annotations                                     │
│    - Apply same transforms to bounding boxes                │
│    - Create Instances object with gt_boxes, gt_classes      │
│                                                               │
│ 4. Apply Strong Augmentation (for student)                   │
│    - Color jitter, blur, etc.                               │
│    - Creates second augmented view                          │
│                                                               │
│ 5. Output: Two Tensor Images                                │
│    - image_q (strong aug) shape: [3, H, W]                  │
│    - image_k (weak aug) shape: [3, H, W]                    │
│    - Both normalized to [0, 255] range                      │
└─────────────────────────────────────────────────────────────┘
    ↓
Returns: (dataset_dict_strong, dataset_dict_weak)
```

### 2. Training Loop Flow

**Main Loop**: `DINOTeacherTrainer.run_step_full_semisup()`

```
Data Batch from DataLoader
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Batch Structure:                                             │
│ - label_data_q:   Source images (strong aug) + GT           │
│ - label_data_k:   Source images (weak aug) + GT             │
│ - unlabel_data_q: Target images (strong aug) - no GT        │
│ - unlabel_data_k: Target images (weak aug) - no GT          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase A: Burn-in Stage (iter < BURN_UP_STEP)                │
│                                                               │
│   Source Images → Student Model                              │
│       ↓                                                       │
│   Supervised Loss (Cross-Entropy + Regression)               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase B: Semi-Supervised Training (iter >= BURN_UP_STEP)    │
│                                                               │
│ Step 1: Generate/Load Pseudo-Labels                          │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Option A: Use Pre-generated DINO Labels              │   │
│   │   unlabel_data → Load from pickle file               │   │
│   │                → Apply transforms                     │   │
│   │                → Threshold by confidence              │   │
│   └─────────────────────────────────────────────────────┘   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Option B: Generate with Mean Teacher (EMA model)     │   │
│   │   unlabel_data_k → Teacher Model (weak aug)          │   │
│   │                  → RPN proposals                      │   │
│   │                  → ROI Head predictions               │   │
│   │                  → Threshold by confidence            │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                               │
│ Step 2: Train on Source Domain                               │
│   label_data (q+k) → Student Model                           │
│                    → Supervised Loss                          │
│                                                               │
│ Step 3: Train on Target Domain                               │
│   unlabel_data_q + pseudo_labels → Student Model             │
│                                  → Unsupervised Loss          │
│                                                               │
│ Step 4: Feature Alignment (if enabled)                       │
│   ┌─────────────────────────────────────────────────────┐   │
│   │ Source Images:                                        │   │
│   │   label_data → DINO Teacher (frozen ViT)             │   │
│   │              → ViT features [B, D, H', W']            │   │
│   │   label_data → Student Backbone                       │   │
│   │              → CNN features [B, C, H, W]              │   │
│   │              → Alignment Head (MLP)                   │   │
│   │              → Projected features [B, D, H', W']      │   │
│   │   Alignment Loss = 1 - cosine_similarity              │   │
│   │                                                        │   │
│   │ Target Images (after iter > ALIGN_TARGET_START):     │   │
│   │   unlabel_data → DINO Teacher + Student               │   │
│   │                → Same alignment process                │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                               │
│ Step 5: Update Teacher Model (EMA)                           │
│   teacher_params = α * teacher_params + (1-α) * student_params │
│   where α = EMA_KEEP_RATE (e.g., 0.9996)                    │
└─────────────────────────────────────────────────────────────┘
    ↓
Total Loss = supervised_loss + λ₁ * unsupervised_loss 
           + λ₂ * alignment_loss + λ₃ * alignment_loss_target
    ↓
Backward() → Optimizer.step()
```

### 3. Image Flow Through Models

#### A. DINO ViT Labeller (Phase 1)

```
Input Image [3, H, W]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ DinoVitFeatureExtractor (build_dino.py)                      │
│                                                               │
│ 1. Preprocessing                                              │
│    BGR → RGB conversion: x = x[:,[2,1,0],:,:]                │
│    Normalize with ImageNet stats                             │
│    - mean: [123.675, 116.280, 103.530]                       │
│    - std: [58.395, 57.120, 57.375]                           │
│                                                               │
│ 2. Vision Transformer Encoding                               │
│    x → ViT encoder (dinov2_vitl14 or vitg14)                 │
│      → Patch embedding (14×14 patches)                       │
│      → Transformer blocks (self-attention)                   │
│      → Output: [B, num_patches, embed_dim]                   │
│         - ViT-L: embed_dim = 1024                            │
│         - ViT-G: embed_dim = 1536                            │
│                                                               │
│ 3. Reshape to Grid                                            │
│    Remove class token (for DinoV1)                           │
│    Reshape: [B, embed_dim, H/14, W/14]                       │
└─────────────────────────────────────────────────────────────┘
    ↓
Grid Features: [B, 1024, H', W'] where H'=H/14, W'=W/14
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Region Proposal Network (RPN)                                │
│                                                               │
│ Features → Conv layers                                        │
│         → Objectness scores                                   │
│         → Bounding box deltas                                 │
│         → NMS filtering                                       │
│ Output: ~2000 proposals per image                            │
└─────────────────────────────────────────────────────────────┘
    ↓
Proposals: List[Instances] with proposal_boxes, objectness_logits
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ROI Head (SingleScaleROIHeadsPseudoLab)                      │
│                                                               │
│ For each proposal:                                            │
│   1. ROI Align: Extract 7×7 feature patch                    │
│   2. FC layers: 2 fully connected layers                     │
│   3. Classification: Predict class (8 classes)               │
│   4. Box Regression: Refine bounding box                     │
│                                                               │
│ Output per proposal:                                          │
│   - pred_classes: [num_proposals]                            │
│   - scores: [num_proposals]                                  │
│   - pred_boxes: [num_proposals, 4]                           │
└─────────────────────────────────────────────────────────────┘
    ↓
Final Detections: 
  Instances(image_size=(H,W),
           pred_boxes=Boxes([x1,y1,x2,y2]),
           pred_classes=[class_ids],
           scores=[confidence])
    ↓
(For pseudo-label generation):
  Filter by confidence threshold (e.g., 0.8)
  Save to pickle file
```

#### B. Student Model with VGG Backbone (Phase 3)

```
Input Image [3, H, W]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ VGG16 Backbone                                                │
│                                                               │
│ Input [3, H, W]                                              │
│   ↓                                                           │
│ Conv1_1 → Conv1_2 → MaxPool → [64, H/2, W/2]                │
│   ↓                                                           │
│ Conv2_1 → Conv2_2 → MaxPool → [128, H/4, W/4]               │
│   ↓                                                           │
│ Conv3_1 → Conv3_2 → Conv3_3 → MaxPool → [256, H/8, W/8]     │
│   ↓                                                           │
│ Conv4_1 → Conv4_2 → Conv4_3 → MaxPool → [512, H/16, W/16]   │
│   ↓                                                           │
│ Conv5_1 → Conv5_2 → Conv5_3 → [512, H/16, W/16]             │
│                                                               │
│ Output: {"vgg4": features [B, 512, H/16, W/16]}             │
└─────────────────────────────────────────────────────────────┘
    ↓
VGG Features [B, 512, H/16, W/16]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Parallel Paths:                                               │
│                                                               │
│ Path 1: Detection Pipeline                                    │
│   Features → RPN → Proposals → ROI Head → Predictions        │
│   (Same as DINO labeller, but with VGG features)             │
│                                                               │
│ Path 2: Feature Alignment (if enabled)                       │
│   VGG Features [B, 512, H/16, W/16]                          │
│       ↓                                                       │
│   Alignment Head (MLP):                                       │
│     Conv2d(512 → 1024) → ReLU                                │
│     Conv2d(1024 → 768)                                        │
│       ↓                                                       │
│   Interpolate to DINO size: [B, 768, H', W']                 │
│   L2 Normalize                                                │
│       ↓                                                       │
│   Compare with DINO Teacher Features [B, 768, H', W']        │
│   Loss = 1 - cosine_similarity                                │
└─────────────────────────────────────────────────────────────┘
```

### 4. Cross-Domain Feature Alignment

```
Source/Target Image
    ↓
┌──────────────────────────────┬──────────────────────────────┐
│ DINO Teacher (Frozen)        │ Student CNN (Training)        │
│                              │                               │
│ Image [3, H, W]              │ Image [3, H, W]              │
│   ↓                          │   ↓                          │
│ ViT Encoder                  │ VGG/ResNet Backbone          │
│   ↓                          │   ↓                          │
│ Features: [B, 768, 37, 68]   │ Features: [B, 512, 34, 62]   │
│   (for 518×952 input)        │   (for 550×1024 input)       │
│   ↓                          │   ↓                          │
│ L2 Normalize                 │ Alignment MLP Head           │
│                              │   ↓                          │
│                              │ Interpolate + L2 Normalize   │
│   ↓                          │   ↓                          │
│ Teacher Features             │ Student Features             │
│ [B, 768, 37, 68]             │ [B, 768, 37, 68]             │
└──────────────────────────────┴──────────────────────────────┘
    ↓                              ↓
    └──────────────┬───────────────┘
                   ↓
         Cosine Similarity Loss
         loss = 1 - (teacher · student)
                   ↓
         Alignment forces student features
         to match DINO's domain-invariant features
```

### 5. Pseudo-Label Generation Flow

```
Target Domain Image (e.g., BDD100k)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Load Trained DINO Labeller                                   │
│   - Checkpoint: model_final.pth                              │
│   - Backbone: DINO ViT-L or ViT-G                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Inference (gen_labels.py)                                    │
│                                                               │
│ For each image in target domain:                             │
│   1. Image → DINO ViT → Grid Features                       │
│   2. Grid Features → RPN → Proposals                         │
│   3. Proposals → ROI Head → Predictions                      │
│   4. Filter by confidence threshold (e.g., 0.8)              │
│   5. Create Instances object:                                │
│      - pred_boxes: [N, 4]                                    │
│      - scores: [N]                                           │
│      - pred_classes: [N]                                     │
│                                                               │
│ Save to pickle file:                                          │
│   [{                                                          │
│     'image_id': int,                                         │
│     'file_name': str,                                        │
│     'height': int,                                           │
│     'width': int,                                            │
│     'instances_dino': Instances                              │
│   }, ...]                                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
Pseudo-labels stored: 
  output/dino_label/predictions/BDD_day_train_dino_anno_vitl.pkl
    ↓
Used in Student Training (Phase 3):
  - Loaded at startup
  - Applied with data augmentation transforms
  - Used as supervision for target domain
```

### 6. Loss Functions

```
Total Loss = Σ weighted losses

┌─────────────────────────────────────────────────────────────┐
│ 1. Supervised Losses (Source Domain)                         │
│    - loss_cls: Cross-entropy for classification              │
│    - loss_box_reg: Smooth L1 for box regression             │
│    - loss_rpn_cls: RPN objectness loss                       │
│    - loss_rpn_loc: RPN localization loss                     │
│    Weight: 1.0                                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 2. Unsupervised Losses (Target Domain with Pseudo-labels)   │
│    - loss_cls_pseudo: Classification loss                    │
│    - loss_box_reg_pseudo: Box regression (typically 0)       │
│    - loss_rpn_cls_pseudo: RPN losses                         │
│    Weight: UNSUP_LOSS_WEIGHT (e.g., 1.0)                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 3. Alignment Losses (Cross-domain)                           │
│    - loss_align: Source domain alignment                     │
│      Weight: FEATURE_ALIGN_LOSS_WEIGHT (e.g., 1.0)          │
│    - loss_align_target: Target domain alignment              │
│      Weight: FEATURE_ALIGN_LOSS_WEIGHT_TARGET (e.g., 1.0)   │
│    Formula: 1 - cosine_similarity(student_feat, dino_feat)   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 4. Optional: Adversarial Domain Loss                         │
│    - loss_D_img_s: Discriminator loss for source            │
│    - loss_D_img_t: Discriminator loss for target            │
│    Weight: DIS_LOSS_WEIGHT (default: 0.0)                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Components Deep Dive

### Component 1: DatasetMapper

**File**: `dinoteacher/data/dataset_mapper.py`

**Purpose**: Transforms raw dataset dictionaries into model inputs

**Key Features**:
- Creates two views per image (strong and weak augmentation)
- Maintains transform information (`tf_data`) for pseudo-label alignment
- Handles multiple datasets (Cityscapes, BDD100k, ACDC)
- Ensures image dimensions are multiples of patch size for ViT

**Augmentation Pipeline**:
1. **Weak Augmentation**:
   - Resize to patch-size multiple (e.g., 518×952 for patch_size=14)
   - Random horizontal flip
   - Random crop and pad (maintains original size)

2. **Strong Augmentation**:
   - Color jitter
   - Gaussian blur
   - Grayscale conversion (random)
   - All applied via PIL transforms

### Component 2: DINO Feature Extractor

**File**: `dinoteacher/engine/build_dino.py`

**Purpose**: Extract domain-invariant features using pre-trained DINO models

**Models Supported**:
- DINO v1: `dino_vitb8`, `dino_vitb16`
- DINO v2: `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`

**Key Operations**:
1. Load pre-trained weights from `weights/` directory
2. Convert BGR to RGB (Detectron2 default is BGR)
3. Normalize with ImageNet statistics
4. Extract intermediate layer features
5. Reshape patch tokens to spatial grid
6. Optional: L2 normalization

**Feature Dimensions**:
- Input: [B, 3, H, W] where H, W are multiples of patch_size
- Output: [B, embed_dim, H/patch_size, W/patch_size]
  - ViT-S: embed_dim = 384
  - ViT-B: embed_dim = 768
  - ViT-L: embed_dim = 1024
  - ViT-G: embed_dim = 1536

### Component 3: Teacher-Student Alignment

**File**: `dinoteacher/engine/align_head.py`

**Purpose**: Align student CNN features with DINO teacher features

**Alignment Head Types**:

1. **MLP** (default):
   ```
   Conv2d(student_dim → proj_dim) → ReLU
   Conv2d(proj_dim → teacher_dim)
   ```

2. **MLP3**:
   ```
   Conv2d(student_dim → proj_dim) → ReLU
   Conv2d(proj_dim → proj_dim) → ReLU
   Conv2d(proj_dim → teacher_dim)
   ```

3. **Linear**:
   ```
   Conv2d(student_dim → teacher_dim)
   ```

4. **Attention**:
   ```
   Multi-head attention → Conv2d projection
   ```

**Alignment Loss**:
- With normalization: `loss = 1 - cosine_similarity`
- Without normalization: `loss = L2_norm / 100`

### Component 4: Training Stages

**File**: `dinoteacher/engine/trainer.py`

**Three Training Stages**:

1. **Burn-in (iter < BURN_UP_STEP)**:
   - Train on source domain only with ground truth
   - Establish baseline performance
   - Typical duration: 20,000 iterations

2. **Source + Target Alignment (BURN_UP_STEP ≤ iter < ALIGN_TARGET_START)**:
   - Continue training on source domain
   - Add feature alignment on source domain
   - Teacher model updated via EMA

3. **Full Training (iter ≥ ALIGN_TARGET_START)**:
   - Train on source (GT) + target (pseudo-labels)
   - Feature alignment on both domains
   - Full semi-supervised learning

## Configuration Files

### For DINO Labeller Training (`vit_labeller.yaml`)

```yaml
MODEL:
  BACKBONE:
    NAME: "build_dino_vit_backbone"  # Use DINO ViT
  ROI_HEADS:
    NUM_CLASSES: 8                    # 8 object classes

SOLVER:
  MAX_ITER: 40000
  BASE_LR: 0.01
  IMG_PER_BATCH_LABEL: 8

DATASETS:
  TRAIN_LABEL: ("cityscapes_fine_instance_seg_train",)
  TEST: ("cityscapes_val", "BDD_day_val")

SEMISUPNET:
  DINO_BBONE_MODEL: "dinov2_vitl14"   # ViT-L backbone
  BBOX_THRESHOLD: 0.8                  # Confidence threshold
  BURN_UP_STEP: 100000                 # No semi-supervised learning
```

### For Student Training (`vgg_city2bdd.yaml`)

```yaml
MODEL:
  BACKBONE:
    NAME: "build_vgg_backbone"        # Use VGG16

DATASETS:
  TRAIN_LABEL: ("cityscapes_fine_instance_seg_train",)
  TRAIN_UNLABEL: ("BDD_day_train",)  # Target domain

SEMISUPNET:
  USE_FEATURE_ALIGN: True
  FEATURE_ALIGN_LAYER: "vgg4"
  ALIGN_MODEL: "dinov2_vitb14"        # DINO for alignment
  FEATURE_ALIGN_TARGET_START: 5000
  LABELER_TARGET_PSEUDOGT: "path/to/pseudo_labels.pkl"
```

## Usage Examples

### 1. Train DINO Labeller

```bash
python train_net.py \
  --num-gpus 2 \
  --config configs/vit_labeller.yaml \
  OUTPUT_DIR output/dino_label/vitl \
  SOLVER.IMG_PER_BATCH_LABEL 8 \
  DATASETS.TEST '("cityscapes_val","BDD_day_val")' \
  SEMISUPNET.DINO_BBONE_MODEL dinov2_vitl14
```

**What happens**:
- Images loaded from Cityscapes training set
- DINO ViT-L backbone extracts features
- RPN + ROI Head trained for detection
- Model checkpoints saved every 5000 iterations
- Evaluation on validation sets

### 2. Generate Pseudo-Labels

```bash
python train_net.py \
  --num-gpus 1 \
  --resume \
  --gen-labels \
  --config configs/vit_labeller.yaml \
  OUTPUT_DIR output/dino_label/vitl \
  DATASETS.TEST '("BDD_day_train",)' \
  SEMISUPNET.DINO_BBONE_MODEL dinov2_vitl14
```

**What happens**:
- Load trained labeller from `output/dino_label/vitl/model_final.pth`
- Run inference on BDD100k training set
- Filter predictions by confidence threshold (0.8)
- Save to `output/dino_label/vitl/predictions/BDD_day_train_dino_anno_vitl.pkl`

### 3. Train Student Model

```bash
python train_net.py \
  --num-gpus 2 \
  --config configs/vgg_city2bdd.yaml \
  SEMISUPNET.LABELER_TARGET_PSEUDOGT output/dino_label/vitl/predictions/BDD_day_train_dino_anno_vitl.pkl
```

**What happens**:
- Load VGG16 backbone
- Create DINO ViT-B for feature alignment (frozen)
- Train with:
  - Source domain (Cityscapes) ground truth
  - Target domain (BDD100k) pseudo-labels
  - Feature alignment losses on both domains
- Teacher model updated via EMA
- Model checkpoints saved every 5000 iterations

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DINO LABELLER TRAINING                              │
│                                                               │
│ Cityscapes Images + GT                                        │
│         ↓                                                     │
│ DatasetMapper (augmentation)                                  │
│         ↓                                                     │
│ DINO ViT Backbone → RPN → ROI Head                          │
│         ↓                                                     │
│ Trained DINO Labeller (model_final.pth)                     │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: PSEUDO-LABEL GENERATION                             │
│                                                               │
│ BDD100k Images (no labels)                                   │
│         ↓                                                     │
│ Trained DINO Labeller                                        │
│         ↓                                                     │
│ Predictions → Filter by confidence                           │
│         ↓                                                     │
│ Pseudo-labels (BDD_day_train_dino_anno_vitl.pkl)           │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: STUDENT MODEL TRAINING                              │
│                                                               │
│ Source: Cityscapes + GT  │  Target: BDD100k + Pseudo-labels │
│         ↓                │          ↓                        │
│         DatasetMapper (strong + weak augmentation)           │
│         ↓                                                     │
│    ┌────────────────┬────────────────────────┐              │
│    │ Student Model  │  DINO Teacher (frozen) │              │
│    │ (VGG/ResNet)   │  (ViT-B for align)     │              │
│    └────────────────┴────────────────────────┘              │
│         ↓                                                     │
│    Supervised Loss + Unsupervised Loss + Alignment Loss      │
│         ↓                                                     │
│    Trained Student Detector                                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Why DINO ViT as Labeller?**
   - Self-supervised pre-training on large-scale data
   - Domain-invariant features
   - High-quality pseudo-labels across domains

2. **Why Feature Alignment?**
   - Bridges the gap between source and target domains
   - Forces student to learn domain-invariant representations
   - Improves generalization to target domain

3. **Why EMA Teacher?**
   - Provides more stable pseudo-labels
   - Reduces noise in self-training
   - Standard practice in semi-supervised learning

4. **Why Two Augmentation Views?**
   - Strong augmentation for student (robustness)
   - Weak augmentation for teacher (stability)
   - Consistency regularization between views

## Performance Characteristics

### DINO Labellers (mAP@0.5)

| Backbone | Cityscapes | Foggy Cityscapes | BDD100k |
|----------|-----------|------------------|---------|
| ViT-L    | 61.3      | 54.6             | 45.7    |
| ViT-G    | 64.3      | 58.8             | 51.1    |

### Student Models (mAP@0.5)

| Target Domain    | Student Backbone | Labeller | Align Teacher | mAP@0.5 |
|-----------------|------------------|----------|---------------|---------|
| Foggy Cityscapes | VGG16           | ViT-G    | ViT-B         | 55.4    |
| BDD100k         | VGG16           | ViT-G    | ViT-B         | 47.8    |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `SOLVER.IMG_PER_BATCH_LABEL`
   - Reduce `INPUT.MIN_SIZE_TRAIN`
   - Use smaller ViT model (ViT-B instead of ViT-L)

2. **Image Size Not Divisible by Patch Size**
   - Ensure images are resized to multiples of patch_size (14 or 16)
   - Check `RandomCropAndPad` augmentation

3. **Pseudo-labels Not Found**
   - Verify pseudo-label file path in config
   - Check that pseudo-label generation completed successfully

4. **Poor Performance**
   - Check DINO labeller quality first
   - Verify pseudo-label confidence threshold
   - Ensure feature alignment is enabled
   - Check learning rate and training iterations

## References

- **Paper**: "Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection" (CVPR 2025)
- **Base Framework**: Adaptive Teacher (https://github.com/facebookresearch/adaptive_teacher)
- **DINO**: Self-supervised Vision Transformers (https://github.com/facebookresearch/dino)
- **Detectron2**: Facebook AI Research's detection framework

## Summary

The DINO Teacher framework processes images through three main phases:

1. **Training a DINO ViT labeller** on source domain to generate high-quality detections
2. **Generating pseudo-labels** for target domain using the trained labeller
3. **Training a student detector** with both source ground truth and target pseudo-labels, while aligning features with a frozen DINO teacher

The key innovation is leveraging vision foundation models (DINO) both as a pseudo-labeller and as an alignment teacher, enabling effective domain adaptation without target domain annotations.
