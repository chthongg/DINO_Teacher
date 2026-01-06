# DINO Teacher: Quick Reference Guide

## 🚀 Quick Start

### Installation
```bash
# See INSTALL.md for full installation instructions
pip install -r requirements.txt
# Download pre-trained DINO weights to weights/ directory
```

### Train DINO Labeller (Phase 1)
```bash
python train_net.py \
  --num-gpus 2 \
  --config configs/vit_labeller.yaml \
  OUTPUT_DIR output/dino_label/vitl \
  SEMISUPNET.DINO_BBONE_MODEL dinov2_vitl14
```

### Generate Pseudo-Labels (Phase 2)
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

### Train Student Model (Phase 3)
```bash
python train_net.py \
  --num-gpus 2 \
  --config configs/vgg_city2bdd.yaml \
  SEMISUPNET.LABELER_TARGET_PSEUDOGT output/dino_label/vitl/predictions/BDD_day_train_dino_anno_vitl.pkl
```

## 📁 Repository Structure

```
DINO_Teacher/
├── train_net.py                    # Main entry point
├── configs/                        # YAML configuration files
│   ├── vit_labeller.yaml          # DINO labeller config
│   └── vgg_city2bdd.yaml          # Student model config
├── dinoteacher/                    # Core implementation
│   ├── engine/
│   │   ├── trainer.py             # Main training loop
│   │   ├── gen_labels.py          # Pseudo-label generation
│   │   ├── build_dino.py          # DINO ViT feature extractor
│   │   └── align_head.py          # Feature alignment module
│   ├── modeling/
│   │   ├── meta_arch/
│   │   │   ├── dino_vit.py        # DINO ViT backbone wrapper
│   │   │   └── rcnn.py            # Modified RCNN architecture
│   │   └── roi_heads/             # ROI head implementations
│   └── data/
│       ├── dataset_mapper.py      # Data loading & augmentation
│       └── datasets/              # Dataset registration
├── adapteacher/                    # Base Adaptive Teacher code
├── dinov2/                        # DINO v2 models
└── weights/                       # Pre-trained model weights (download separately)
```

## 🔄 Image Processing Pipeline

### Input → Output Flow

```
Raw Image (e.g., 1024×2048 JPG)
    ↓
┌─────────────────────────────────────┐
│ 1. DatasetMapper                    │
│    - Read image                     │
│    - Resize to patch multiple       │
│    - Augmentation (weak + strong)   │
│    - Normalize                      │
└─────────────────────────────────────┘
    ↓
Tensor [3, H, W] (e.g., 518×952)
    ↓
┌─────────────────────────────────────┐
│ 2. Backbone Feature Extraction      │
│    - DINO ViT: → [B, 1024, 37, 68] │
│    - VGG16: → [B, 512, H/16, W/16] │
└─────────────────────────────────────┘
    ↓
Feature Maps
    ↓
┌─────────────────────────────────────┐
│ 3. Region Proposal Network (RPN)    │
│    - Generate ~2000 proposals       │
│    - Objectness scoring             │
│    - NMS filtering                  │
└─────────────────────────────────────┘
    ↓
Proposals (bounding boxes)
    ↓
┌─────────────────────────────────────┐
│ 4. ROI Head                         │
│    - ROI Align (7×7 features)       │
│    - Classification                 │
│    - Box regression                 │
└─────────────────────────────────────┘
    ↓
Final Detections:
- Bounding boxes [N, 4]
- Class labels [N]
- Confidence scores [N]
```

## 🧠 Model Architecture

### DINO ViT Labeller
```
Input Image [3, H, W]
    ↓
DINO ViT Encoder (frozen/fine-tuned)
├── Patch Embedding (14×14 patches)
├── Transformer Blocks (self-attention)
└── Output: [B, embed_dim, H/14, W/14]
    ↓
RPN (trainable)
├── Objectness prediction
└── Bounding box proposals
    ↓
ROI Head (trainable)
├── ROI Align
├── FC layers
├── Classification (8 classes)
└── Box regression
    ↓
Detections
```

### Student Model with Alignment
```
Input Image [3, H, W]
    ↓
┌────────────────────┬─────────────────────┐
│ Student Path       │ Teacher Path        │
│                    │ (frozen)            │
│ VGG/ResNet         │ DINO ViT            │
│     ↓              │     ↓               │
│ Features           │ Features            │
│ [B, 512, H/16, W/16] [B, 768, H/14, W/14]
│     ↓              │     ↓               │
│ Alignment Head     │ L2 Normalize        │
│     ↓              │     ↓               │
│ Projected Features │ DINO Features       │
│ [B, 768, H', W']   │ [B, 768, H', W']   │
└────────────────────┴─────────────────────┘
         ↓                    ↓
         └────────┬───────────┘
                  ↓
         Alignment Loss
         (cosine similarity)
```

## 📊 Training Stages

### Stage 1: Burn-in (iter 0 → BURN_UP_STEP)
- **Data**: Source domain only (Cityscapes)
- **Supervision**: Ground truth labels
- **Loss**: Supervised detection loss
- **Goal**: Warm up the student model

### Stage 2: Source Alignment (BURN_UP_STEP → ALIGN_TARGET_START)
- **Data**: Source domain
- **Supervision**: Ground truth + DINO alignment
- **Loss**: Supervised + alignment loss (source)
- **Goal**: Learn domain-invariant features on source

### Stage 3: Full Training (iter ≥ ALIGN_TARGET_START)
- **Data**: Source + Target domains
- **Supervision**: Source GT + Target pseudo-labels + DINO alignment
- **Loss**: Supervised + unsupervised + alignment (source + target)
- **Goal**: Adapt to target domain

## 🎯 Key Hyperparameters

### DINO Labeller Training
```yaml
SOLVER:
  MAX_ITER: 40000              # Training iterations
  BASE_LR: 0.01                # Learning rate
  IMG_PER_BATCH_LABEL: 8       # Batch size

SEMISUPNET:
  DINO_BBONE_MODEL: "dinov2_vitl14"  # ViT-L or ViT-G
  BBOX_THRESHOLD: 0.8          # Confidence threshold for pseudo-labels
  BURN_UP_STEP: 100000         # Disable semi-supervised (source only)
```

### Student Model Training
```yaml
SOLVER:
  MAX_ITER: 60000
  BASE_LR: 0.04
  IMG_PER_BATCH_LABEL: 8       # Source batch size
  IMG_PER_BATCH_UNLABEL: 8     # Target batch size

SEMISUPNET:
  BBOX_THRESHOLD: 0.8          # Pseudo-label threshold
  BURN_UP_STEP: 20000          # Supervised warm-up iterations
  EMA_KEEP_RATE: 0.9996        # Teacher EMA momentum
  UNSUP_LOSS_WEIGHT: 1.0       # Weight for pseudo-label loss
  
  # Feature Alignment
  USE_FEATURE_ALIGN: True
  ALIGN_MODEL: "dinov2_vitb14"  # DINO model for alignment
  FEATURE_ALIGN_TARGET_START: 5000  # When to start target alignment
  FEATURE_ALIGN_LOSS_WEIGHT: 1.0    # Source alignment weight
  FEATURE_ALIGN_LOSS_WEIGHT_TARGET: 1.0  # Target alignment weight
  
  # Pseudo-labels
  LABELER_TARGET_PSEUDOGT: "path/to/pseudo_labels.pkl"
```

## 📝 Loss Functions

```python
# Total loss computation
total_loss = (
    # Source domain (ground truth)
    1.0 * supervised_loss
    
    # Target domain (pseudo-labels)
    + UNSUP_LOSS_WEIGHT * unsupervised_loss
    
    # Feature alignment (source)
    + FEATURE_ALIGN_LOSS_WEIGHT * alignment_loss
    
    # Feature alignment (target)
    + FEATURE_ALIGN_LOSS_WEIGHT_TARGET * alignment_loss_target
)
```

### Breakdown:
- **supervised_loss**: Classification + box regression on source
- **unsupervised_loss**: Classification on target (box regression weight = 0)
- **alignment_loss**: 1 - cosine_similarity(student_feat, dino_feat)

## 🔍 Debugging Tips

### Check Data Loading
```python
# In trainer.py, add breakpoint at line 314
data = next(self._trainer._data_loader_iter)
label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

# Inspect shapes
print(label_data_q[0]['image'].shape)    # Should be [3, H, W]
print(label_data_q[0]['instances'])      # Instances object with gt_boxes
```

### Monitor Training Losses
```bash
# TensorBoard logs saved to OUTPUT_DIR
tensorboard --logdir output/
```

Key metrics to watch:
- `loss_cls`: Classification loss (should decrease)
- `loss_box_reg`: Box regression loss (should decrease)
- `loss_align`: Alignment loss (should decrease to ~0.2-0.4)
- `total_loss`: Overall loss (should decrease steadily)

### Verify Pseudo-Labels
```python
import pickle

with open('output/dino_label/predictions/BDD_day_train_dino_anno_vitl.pkl', 'rb') as f:
    pseudo_labels = pickle.load(f)

# Check first image
print(pseudo_labels[0].keys())  # ['file_name', 'image_id', 'height', 'width', 'instances_dino']
print(pseudo_labels[0]['instances_dino'])  # Instances with pred_boxes, scores, pred_classes
print(f"Number of detections: {len(pseudo_labels[0]['instances_dino'])}")
```

### Common Issues

1. **Out of Memory**
   ```yaml
   # Reduce batch size
   SOLVER:
     IMG_PER_BATCH_LABEL: 4
     IMG_PER_BATCH_UNLABEL: 4
   
   # Or reduce image size
   INPUT:
     MIN_SIZE_TRAIN: (480,)  # Instead of (600,)
   ```

2. **Image Size Mismatch**
   ```yaml
   # Ensure divisible by patch size
   INPUT:
     CROP:
       ENABLED: True
       SIZE: [518, 952]  # Multiples of 14
   SEMISUPNET:
     DINO_PATCH_SIZE: 14
   ```

3. **Pseudo-Labels Not Used**
   ```yaml
   # Check config
   SEMISUPNET:
     LABELER_TARGET_PSEUDOGT: "output/dino_label/predictions/BDD_day_train_dino_anno_vitl.pkl"
   
   # Verify file exists
   # Check iteration is past BURN_UP_STEP
   ```

## 🎓 Understanding the Code

### Key Files to Read

1. **Start Here**: `train_net.py`
   - Entry point
   - Model setup
   - Training mode selection

2. **Training Loop**: `dinoteacher/engine/trainer.py`
   - `DINOTeacherTrainer.__init__()`: Initialization
   - `run_step_full_semisup()`: Main training step (line 310)
   - `_update_teacher_model()`: EMA update (line 570)

3. **Data Loading**: `dinoteacher/data/dataset_mapper.py`
   - `DatasetMapperTwoCropSeparateKeepTf.__call__()`: Data transformation
   - Augmentation pipeline (line 226)

4. **DINO Features**: `dinoteacher/engine/build_dino.py`
   - `DinoVitFeatureExtractor.forward()`: Feature extraction (line 103)
   - Preprocessing and normalization

5. **Feature Alignment**: `dinoteacher/engine/align_head.py`
   - `TeacherStudentAlignHead.forward()`: Projection (line 37)
   - `align_loss()`: Similarity computation (line 48)

### Code Flow for One Training Iteration

```python
# 1. Load batch (dinoteacher/engine/trainer.py:314)
data = next(self._trainer._data_loader_iter)
label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

# 2. Update teacher model via EMA (line 321)
if self.iter % TEACHER_UPDATE_ITER == 0:
    self._update_teacher_model(keep_rate=EMA_KEEP_RATE)

# 3. Generate or load pseudo-labels (line 361-401)
if use_DT_labels:
    # Load from pickle file
    instances = [self.dino_pseudogt[x['image_id']]['instances_dino'] 
                 for x in unlabel_data_q]
else:
    # Generate with teacher model
    _, _, proposals_roih_unsup_k, _ = self.model_teacher(unlabel_data_k)
    pseudo_proposals = self.process_pseudo_label(proposals_roih_unsup_k, threshold)

# 4. Train student on source (line 416)
record_all_label_data, _, _, _ = self.model(all_label_data, branch="supervised")

# 5. Train student on target (line 422)
record_all_unlabel_data, _, _, _ = self.model(all_unlabel_data, branch="supervised_target")

# 6. Feature alignment (line 454)
if self.use_feature_align:
    teacher_feat = self.model.align_teacher(all_label_data)
    student_feat = self.model.align_student_head(self.student_align_feat['supervised'])
    align_loss = self.model.align_student_head.align_loss(student_feat, teacher_feat)

# 7. Compute total loss (line 496)
losses = sum(loss_dict.values())

# 8. Backpropagation (line 526)
self.optimizer.zero_grad()
losses.backward()
self.optimizer.step()
```

## 📈 Expected Results

### DINO Labeller Performance (mAP@0.5)

| Model  | Source (CS) | Target 1 (FCS) | Target 2 (BDD) |
|--------|-------------|----------------|----------------|
| ViT-L  | 61.3        | 54.6           | 45.7           |
| ViT-G  | 64.3        | 58.8           | 51.1           |

CS = Cityscapes, FCS = Foggy Cityscapes, BDD = BDD100k

### Student Model Performance (mAP@0.5)

| Adaptation     | Backbone | Labeller | Final mAP |
|----------------|----------|----------|-----------|
| CS → FCS       | VGG16    | ViT-G    | 55.4      |
| CS → BDD       | VGG16    | ViT-G    | 47.8      |

Improvements over baseline:
- Foggy Cityscapes: +15-20% over source-only
- BDD100k: +10-15% over source-only

## 🔗 Key Concepts

### Mean Teacher (EMA)
The teacher model is an exponential moving average of the student:
```python
θ_teacher = α * θ_teacher + (1 - α) * θ_student
```
Where α = 0.9996 (EMA_KEEP_RATE)

### Pseudo-Labeling
1. Teacher generates predictions on unlabeled data
2. Filter by confidence threshold (e.g., 0.8)
3. Use as "ground truth" for student training

### Feature Alignment
Forces student CNN features to match DINO ViT features:
- Encourages domain-invariant representations
- Bridges source-target domain gap
- Uses cosine similarity loss

### Two-Stage Training
1. **Source-only**: Establish baseline
2. **Source + Target**: Semi-supervised adaptation

## 🛠️ Customization

### Add New Dataset
1. Register in `dinoteacher/data/datasets/builtin.py`
2. Add to config `DATASETS.TRAIN_UNLABEL`

### Change Backbone
```yaml
MODEL:
  BACKBONE:
    NAME: "build_resnet_backbone"  # Instead of build_vgg_backbone
```

### Adjust Alignment
```yaml
SEMISUPNET:
  ALIGN_HEAD_TYPE: "MLP3"        # MLP, MLP3, linear, attention
  ALIGN_HEAD_PROJ_DIM: 2048      # Projection dimension
  ALIGN_PROJ_GELU: True          # Use GELU activation
```

## 📚 Additional Resources

- **Full Documentation**: See `CODE_FLOW.md` for detailed architecture
- **Paper**: [arXiv:2503.23220](https://arxiv.org/pdf/2503.23220)
- **Base Framework**: [Adaptive Teacher](https://github.com/facebookresearch/adaptive_teacher)
- **DINO Models**: [DINO v2](https://github.com/facebookresearch/dinov2)

## 🤝 Contributing

For questions or issues:
1. Check existing issues on GitHub
2. Refer to INSTALL.md for setup problems
3. See CODE_FLOW.md for architecture details
