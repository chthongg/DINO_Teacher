# DINO Teacher: Three-Phase Pipeline

```
╔════════════════════════════════════════════════════════════════════════════╗
║                           PHASE 1: DINO LABELLER TRAINING                  ║
╚════════════════════════════════════════════════════════════════════════════╝

    Source Domain (Cityscapes)
    Images + Ground Truth Labels
           │
           │ DataLoader + Augmentation
           ▼
    ┌──────────────────────┐
    │   DINO ViT Backbone  │  ← Pre-trained weights
    │   (ViT-L or ViT-G)   │     (dinov2_vitl14.pth)
    └──────────────────────┘
           │
           │ Grid Features [B, 1024, H/14, W/14]
           ▼
    ┌──────────────────────┐
    │    RPN + ROI Head    │
    │     (trainable)      │
    └──────────────────────┘
           │
           │ Supervised Training
           │ Loss: Classification + Box Regression
           ▼
    ╔══════════════════════╗
    ║  Trained Labeller    ║
    ║  model_final.pth     ║
    ╚══════════════════════╝


╔════════════════════════════════════════════════════════════════════════════╗
║                      PHASE 2: PSEUDO-LABEL GENERATION                      ║
╚════════════════════════════════════════════════════════════════════════════╝

    Target Domain (BDD100k / Foggy Cityscapes)
    Images (NO labels)
           │
           │ DataLoader
           ▼
    ┌──────────────────────┐
    │  Trained Labeller    │  ← Load from Phase 1
    │  (inference mode)    │
    └──────────────────────┘
           │
           │ Predictions: boxes, classes, scores
           ▼
    ┌──────────────────────┐
    │  Confidence Filter   │
    │  threshold = 0.8     │
    └──────────────────────┘
           │
           │ High-confidence predictions
           ▼
    ╔══════════════════════╗
    ║   Pseudo-labels      ║
    ║   .pkl file          ║
    ╚══════════════════════╝


╔════════════════════════════════════════════════════════════════════════════╗
║                       PHASE 3: STUDENT MODEL TRAINING                      ║
╚════════════════════════════════════════════════════════════════════════════╝

    ┌─────────────────────────┐      ┌────────────────────────┐
    │  Source Domain (CS)     │      │  Target Domain (BDD)   │
    │  Images + GT Labels     │      │  Images + Pseudo-labels│
    └─────────────────────────┘      └────────────────────────┘
               │                                │
               │ Strong + Weak Aug              │ Strong + Weak Aug
               ▼                                ▼
    ┌───────────────────────────────────────────────────────┐
    │                   Student Model                        │
    │               (VGG16 or ResNet50)                      │
    └───────────────────────────────────────────────────────┘
               │                                │
               │ Features                       │ Features
               │ [B, 512, H/16, W/16]           │ [B, 512, H/16, W/16]
               ▼                                ▼
    ┌───────────────────────────────────────────────────────┐
    │              Alignment Projection Head                 │
    │                    (MLP Layers)                        │
    └───────────────────────────────────────────────────────┘
               │                                │
               │ Projected Features             │ Projected Features
               ▼                                ▼
    ┌───────────────────────────────────────────────────────┐
    │         Compare with DINO Teacher Features             │
    │           (frozen DINO ViT-B for alignment)            │
    └───────────────────────────────────────────────────────┘
               │                                │
               ▼                                ▼
    ┌────────────────────┐          ┌──────────────────────┐
    │  Supervised Loss   │          │  Unsupervised Loss   │
    │  + Alignment Loss  │          │  + Alignment Loss    │
    └────────────────────┘          └──────────────────────┘
               │                                │
               └────────────┬───────────────────┘
                            ▼
                    ┌───────────────┐
                    │  Total Loss   │
                    └───────────────┘
                            │
                            │ Backpropagation
                            ▼
                    ┌───────────────┐
                    │ Update Student│
                    └───────────────┘
                            │
                            │ EMA (α=0.9996)
                            ▼
                    ┌───────────────┐
                    │ Update Teacher│
                    └───────────────┘
                            │
                            ▼
                    ╔═══════════════╗
                    ║ Trained Model ║
                    ║ (adapted to   ║
                    ║ target domain)║
                    ╚═══════════════╝
```

## Key Components Explained

### DINO ViT Backbone
- **Purpose**: Extract domain-invariant features
- **Models**: ViT-L (1024-dim) or ViT-G (1536-dim)
- **Pre-training**: Self-supervised on large-scale data
- **Output**: Grid of patch features [B, embed_dim, H/14, W/14]

### Student Model (VGG/ResNet)
- **Purpose**: Lightweight detector for deployment
- **Backbone**: VGG16 or ResNet50
- **Training**: Supervised (source) + Unsupervised (target)
- **Alignment**: Features aligned with frozen DINO

### Mean Teacher (EMA)
- **Purpose**: Generate stable pseudo-labels
- **Update**: θ_teacher = 0.9996 * θ_teacher + 0.0004 * θ_student
- **Usage**: Used for pseudo-label generation in Phase 3

### Feature Alignment
- **Purpose**: Bridge domain gap
- **Method**: Project student features to DINO feature space
- **Loss**: 1 - cosine_similarity(student_feat, dino_feat)
- **Applied to**: Both source and target domains

## Training Timeline

```
Iteration:  0        20k       25k       30k       40k       50k       60k
            │─────────│─────────│─────────│─────────│─────────│─────────│
Phase 1:    │═════════════════════════════════════════│
            │  DINO Labeller Training (40k iters)    │
            
Phase 2:                                             │═════════│
                                                     │Generate │
                                                     │ P-Labels│

Phase 3:    │───────────────────│─────────────────────────────────────────│
            │   Burn-in         │  Src Align  │   Full Training          │
            │ (Src GT only)     │ (Src GT +   │ (Src GT + Tgt PL +      │
            │                   │  Alignment) │  Src+Tgt Alignment)     │
            0                  20k           25k                        60k
            
Legend:
─── Source domain supervised training
═══ DINO labeller training  
PL = Pseudo-labels
Src = Source domain
Tgt = Target domain
GT = Ground truth
```

## Data Flow Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    Single Training Iteration                    │
└────────────────────────────────────────────────────────────────┘

Step 1: Load Batch
    ├─ Source (Cityscapes): 8 images with GT, strong + weak aug
    └─ Target (BDD100k): 8 images with pseudo-labels, strong + weak aug

Step 2: Update Teacher (every N iterations)
    └─ θ_teacher ← EMA(θ_teacher, θ_student)

Step 3: Forward Pass - Source Domain
    ├─ Student(source_images) → predictions
    ├─ Compute supervised loss vs GT
    ├─ DINO Teacher(source_images) → teacher_features (frozen)
    ├─ Extract student features → project → student_features_proj
    └─ Compute alignment_loss = 1 - cosine_sim(student_proj, teacher)

Step 4: Forward Pass - Target Domain
    ├─ Student(target_images) → predictions  
    ├─ Compute unsupervised loss vs pseudo-labels
    ├─ DINO Teacher(target_images) → teacher_features (frozen)
    ├─ Extract student features → project → student_features_proj
    └─ Compute alignment_loss_target = 1 - cosine_sim(student_proj, teacher)

Step 5: Total Loss
    └─ total = supervised_loss 
             + λ₁ * unsupervised_loss 
             + λ₂ * alignment_loss
             + λ₃ * alignment_loss_target

Step 6: Backward & Update
    ├─ total_loss.backward()
    └─ optimizer.step()
```

## Model Sizes & Performance

### DINO Labellers (trained on Cityscapes)

| Model    | Params | Feature Dim | Cityscapes | Foggy CS | BDD100k |
|----------|--------|-------------|------------|----------|---------|
| ViT-L/14 | 307M   | 1024        | 61.3       | 54.6     | 45.7    |
| ViT-G/14 | 1.1B   | 1536        | 64.3       | 58.8     | 51.1    |

### Student Models (after adaptation)

| Backbone | Params | Target Domain    | mAP@0.5 | Improvement |
|----------|--------|------------------|---------|-------------|
| VGG16    | 138M   | Foggy Cityscapes | 55.4    | +15-20%     |
| VGG16    | 138M   | BDD100k          | 47.8    | +10-15%     |

## Hardware Requirements

### DINO Labeller Training
- **GPUs**: 2× NVIDIA GPU (≥16GB VRAM each)
- **Time**: ~8-12 hours for 40k iterations
- **Batch Size**: 8 images per GPU

### Pseudo-Label Generation
- **GPUs**: 1× NVIDIA GPU (≥16GB VRAM)
- **Time**: ~1-2 hours for 10k images
- **Speed**: ~5-10 images/sec

### Student Training
- **GPUs**: 2× NVIDIA GPU (≥16GB VRAM each)
- **Time**: ~12-16 hours for 60k iterations
- **Batch Size**: 8 source + 8 target per GPU

## File Outputs

```
output/
├── dino_label/                          # Phase 1 & 2
│   └── vitl/
│       ├── model_final.pth             # Trained labeller (Phase 1)
│       ├── model_0005000.pth           # Checkpoints every 5k iters
│       ├── metrics.json                # Training metrics
│       ├── events.out.tfevents.*       # TensorBoard logs
│       └── predictions/                # Phase 2 outputs
│           ├── BDD_day_train_dino_anno_vitl.pkl
│           └── cityscapes_foggy_train_dino_anno_vitl.pkl
│
└── student_model/                       # Phase 3
    └── vgg_city2bdd/
        ├── model_final.pth             # Final student model
        ├── model_0005000.pth           # Checkpoints
        ├── metrics.json                # Training metrics
        └── events.out.tfevents.*       # TensorBoard logs
```

## Quick Command Reference

```bash
# Phase 1: Train DINO labeller
python train_net.py --num-gpus 2 --config configs/vit_labeller.yaml \
    OUTPUT_DIR output/dino_label/vitl \
    SEMISUPNET.DINO_BBONE_MODEL dinov2_vitl14

# Phase 2: Generate pseudo-labels
python train_net.py --num-gpus 1 --resume --gen-labels \
    --config configs/vit_labeller.yaml \
    OUTPUT_DIR output/dino_label/vitl \
    DATASETS.TEST '("BDD_day_train",)' \
    SEMISUPNET.DINO_BBONE_MODEL dinov2_vitl14

# Phase 3: Train student with adaptation
python train_net.py --num-gpus 2 --config configs/vgg_city2bdd.yaml \
    OUTPUT_DIR output/student_model/vgg_city2bdd \
    SEMISUPNET.LABELER_TARGET_PSEUDOGT output/dino_label/vitl/predictions/BDD_day_train_dino_anno_vitl.pkl

# Evaluate model
python train_net.py --eval-only --config configs/vgg_city2bdd.yaml \
    OUTPUT_DIR output/student_model/vgg_city2bdd \
    MODEL.WEIGHTS output/student_model/vgg_city2bdd/model_final.pth
```

## Citation

```bibtex
@article{lavoie2025large,
  title={Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection},
  author={Lavoie, Marc-Antoine and Mahmoud, Anas and Waslander, Steven L},
  journal={arXiv preprint arXiv:2503.23220},
  year={2025}
}
```
