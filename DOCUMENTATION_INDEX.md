# DINO Teacher Documentation Index

Welcome to the DINO Teacher documentation! This repository implements a domain adaptive object detection framework that uses DINO (self-supervised Vision Transformers) for pseudo-labeling and cross-domain feature alignment.

## 📖 Documentation Overview

We have created comprehensive documentation explaining how the code works and how images flow through the system from beginning to end:

### 🎯 Start Here

1. **[README.md](README.md)**
   - Project overview and quick start
   - Installation links
   - Training commands
   - Results and pre-trained weights

2. **[PIPELINE_VISUAL.md](PIPELINE_VISUAL.md)** ⭐ **RECOMMENDED START**
   - Visual ASCII diagrams of the three-phase pipeline
   - High-level architecture overview
   - Training timeline and data flow
   - Quick command reference
   - Perfect for understanding the big picture

### 📚 Detailed Documentation

3. **[CODE_FLOW.md](CODE_FLOW.md)** ⭐ **MOST COMPREHENSIVE**
   - Complete image flow from input to output
   - Detailed explanation of each processing stage
   - Step-by-step breakdown of training loop
   - Model architectures deep dive
   - Loss functions and training stages
   - ~1000 lines of detailed documentation

4. **[ARCHITECTURE_QUICK_REFERENCE.md](ARCHITECTURE_QUICK_REFERENCE.md)** ⭐ **FOR DAILY USE**
   - Quick reference guide for common tasks
   - Key hyperparameters
   - Debugging tips and common issues
   - Code snippets and examples
   - Perfect companion while working with the code

5. **[INSTALL.md](INSTALL.md)**
   - Detailed installation instructions
   - Dependencies and requirements
   - Environment setup

## 🎓 Learning Path

### For First-Time Users
```
1. Read README.md (5 min)
   ↓
2. Read PIPELINE_VISUAL.md (15 min)
   ↓
3. Skim CODE_FLOW.md sections of interest (30 min)
   ↓
4. Follow INSTALL.md to set up environment (30 min)
   ↓
5. Run Phase 1 training with ARCHITECTURE_QUICK_REFERENCE.md (hands-on)
```

### For Understanding the Architecture
```
1. PIPELINE_VISUAL.md - Visual overview
   ↓
2. CODE_FLOW.md - Section: "High-Level Architecture"
   ↓
3. CODE_FLOW.md - Section: "Complete Image Flow"
   ↓
4. CODE_FLOW.md - Section: "Image Flow Through Models"
```

### For Implementation Details
```
1. CODE_FLOW.md - Section: "Key Components Deep Dive"
   ↓
2. ARCHITECTURE_QUICK_REFERENCE.md - Section: "Understanding the Code"
   ↓
3. Explore source code with documentation as reference
```

### For Training Your Own Model
```
1. ARCHITECTURE_QUICK_REFERENCE.md - Section: "Quick Start"
   ↓
2. INSTALL.md - Set up environment
   ↓
3. PIPELINE_VISUAL.md - Section: "Quick Command Reference"
   ↓
4. ARCHITECTURE_QUICK_REFERENCE.md - Section: "Debugging Tips"
```

## 📊 What Each Document Covers

### Complete Image Flow (CODE_FLOW.md)

This document answers **"How does an image flow from input to output?"**

Covers:
- ✅ Data loading and preprocessing (DatasetMapper)
- ✅ Augmentation pipeline (weak + strong)
- ✅ Feature extraction (DINO ViT and VGG/ResNet)
- ✅ Region proposals (RPN)
- ✅ Detection heads (ROI Head)
- ✅ Feature alignment mechanism
- ✅ Pseudo-label generation and usage
- ✅ Loss computation
- ✅ Training loop iteration
- ✅ Teacher model update (EMA)

**Example sections:**
- "1. Data Loading and Preprocessing" - Shows exact transformations applied
- "2. Training Loop Flow" - Step-by-step iteration walkthrough
- "3. Image Flow Through Models" - Traces image through network layers
- "6. Loss Functions" - Explains all loss terms

### Three-Phase Pipeline (PIPELINE_VISUAL.md)

Visual diagrams showing:
- ✅ Phase 1: DINO Labeller Training
- ✅ Phase 2: Pseudo-Label Generation  
- ✅ Phase 3: Student Model Training
- ✅ Data flow between phases
- ✅ Model components and connections
- ✅ Timeline and training stages

**Perfect for:**
- Getting a quick overview
- Understanding the workflow
- Seeing how phases connect
- Teaching others about the system

### Quick Reference (ARCHITECTURE_QUICK_REFERENCE.md)

Practical guide with:
- ✅ Installation steps
- ✅ Quick start commands
- ✅ Repository structure explanation
- ✅ Common hyperparameters
- ✅ Debugging tips and solutions
- ✅ Key file locations
- ✅ Expected results
- ✅ Code snippets for common tasks

**Perfect for:**
- Day-to-day work
- Troubleshooting issues
- Finding the right hyperparameter
- Understanding error messages

## 🔍 Finding Information

### "How does the code process an image?"
→ **CODE_FLOW.md** - Section: "Complete Image Flow"

### "What are the three training phases?"
→ **PIPELINE_VISUAL.md** - See ASCII diagrams

### "How do I train a model?"
→ **ARCHITECTURE_QUICK_REFERENCE.md** - Section: "Quick Start"

### "What does each file do?"
→ **CODE_FLOW.md** - Section: "Repository Structure"
→ **ARCHITECTURE_QUICK_REFERENCE.md** - Section: "Repository Structure"

### "How does feature alignment work?"
→ **CODE_FLOW.md** - Section: "4. Cross-Domain Feature Alignment"
→ **ARCHITECTURE_QUICK_REFERENCE.md** - Section: "Key Concepts"

### "What are the loss functions?"
→ **CODE_FLOW.md** - Section: "6. Loss Functions"
→ **ARCHITECTURE_QUICK_REFERENCE.md** - Section: "Loss Functions"

### "How do I debug issues?"
→ **ARCHITECTURE_QUICK_REFERENCE.md** - Section: "Debugging Tips"

### "What hyperparameters should I use?"
→ **ARCHITECTURE_QUICK_REFERENCE.md** - Section: "Key Hyperparameters"

### "How does DINO feature extraction work?"
→ **CODE_FLOW.md** - Section: "Component 2: DINO Feature Extractor"

### "How are pseudo-labels generated?"
→ **CODE_FLOW.md** - Section: "5. Pseudo-Label Generation Flow"
→ **PIPELINE_VISUAL.md** - Phase 2 diagram

## 💡 Key Concepts Explained

All documents explain these core concepts:

1. **DINO Vision Transformers**
   - Self-supervised pre-training
   - Domain-invariant features
   - Used for both labeling and alignment

2. **Three-Phase Training**
   - Phase 1: Train DINO labeller on source
   - Phase 2: Generate pseudo-labels for target
   - Phase 3: Train student with adaptation

3. **Mean Teacher (EMA)**
   - Exponential moving average of student
   - Provides stable pseudo-labels
   - θ_teacher = 0.9996 * θ_teacher + 0.0004 * θ_student

4. **Feature Alignment**
   - Projects student features to DINO space
   - Bridges source-target domain gap
   - Uses cosine similarity loss

5. **Pseudo-Labeling**
   - High-quality labels from DINO labeller
   - Confidence threshold filtering (0.8)
   - Used as supervision for target domain

## 📈 Code Statistics

- **Total Documentation**: ~1,700 lines across 5 markdown files
- **Main Documentation**: ~1,000 lines in CODE_FLOW.md
- **Diagrams**: Multiple ASCII art visualizations
- **Code Examples**: Numerous snippets and walkthroughs

## 🛠️ Using the Documentation

### While Coding
Keep **ARCHITECTURE_QUICK_REFERENCE.md** open for:
- Quick command lookup
- Hyperparameter reference
- Debugging tips
- File structure

### While Learning
Read through **CODE_FLOW.md** to understand:
- System architecture
- Data processing pipeline
- Model components
- Training procedure

### While Presenting
Use **PIPELINE_VISUAL.md** to show:
- High-level workflow
- Phase diagrams
- Model architecture
- Data flow

## 🔗 External Resources

- **Paper**: [arXiv:2503.23220](https://arxiv.org/pdf/2503.23220)
- **Adaptive Teacher**: https://github.com/facebookresearch/adaptive_teacher
- **DINO v2**: https://github.com/facebookresearch/dinov2
- **Detectron2**: https://github.com/facebookresearch/detectron2

## 📝 Documentation Features

### Visual Elements
- ✅ ASCII art diagrams
- ✅ Flow charts
- ✅ Box diagrams
- ✅ Timeline visualizations
- ✅ Architecture schematics

### Code Examples
- ✅ Training commands
- ✅ Configuration snippets
- ✅ Python code samples
- ✅ Debug commands
- ✅ File I/O examples

### Explanations
- ✅ Step-by-step walkthroughs
- ✅ Component descriptions
- ✅ Design decisions
- ✅ Performance characteristics
- ✅ Troubleshooting guides

## 🎯 Common Use Cases

### "I want to understand how images are processed"
1. Start with **PIPELINE_VISUAL.md** for overview
2. Deep dive into **CODE_FLOW.md** Section 1-4
3. See specific examples in **ARCHITECTURE_QUICK_REFERENCE.md**

### "I want to train my own model"
1. Read **README.md** for overview
2. Follow **INSTALL.md** for setup
3. Use **ARCHITECTURE_QUICK_REFERENCE.md** for commands
4. Reference **PIPELINE_VISUAL.md** for workflow

### "I want to modify the code"
1. Read **CODE_FLOW.md** Section "Key Components Deep Dive"
2. Use **ARCHITECTURE_QUICK_REFERENCE.md** Section "Understanding the Code"
3. Refer to source files with documentation as guide

### "I'm debugging an error"
1. Check **ARCHITECTURE_QUICK_REFERENCE.md** Section "Debugging Tips"
2. Review **CODE_FLOW.md** for relevant component
3. See **PIPELINE_VISUAL.md** to verify workflow step

## 📊 Document Lengths

| Document | Lines | Purpose | Best For |
|----------|-------|---------|----------|
| CODE_FLOW.md | ~1000 | Complete technical reference | Deep understanding |
| PIPELINE_VISUAL.md | ~350 | Visual pipeline guide | Quick overview |
| ARCHITECTURE_QUICK_REFERENCE.md | ~450 | Daily reference | Practical tasks |
| README.md | ~90 | Project overview | Introduction |
| INSTALL.md | ~120 | Setup guide | Getting started |

## ✅ Documentation Quality

Our documentation:
- ✅ Answers the question "How does code work?"
- ✅ Explains image flow from beginning to end
- ✅ Includes visual diagrams
- ✅ Provides practical examples
- ✅ Covers all major components
- ✅ Includes debugging tips
- ✅ References source code locations
- ✅ Explains design decisions
- ✅ Shows expected outputs
- ✅ Gives performance characteristics

## 🚀 Getting Started

**New to DINO Teacher?**
```bash
# 1. Read this index (you are here!)
# 2. Read the pipeline overview
cat PIPELINE_VISUAL.md

# 3. Set up environment
cat INSTALL.md

# 4. Try Phase 1 training
python train_net.py --num-gpus 2 --config configs/vit_labeller.yaml \
    OUTPUT_DIR output/test_run

# 5. Keep ARCHITECTURE_QUICK_REFERENCE.md open while working!
```

**Already familiar?**
- Use **ARCHITECTURE_QUICK_REFERENCE.md** for daily tasks
- Reference **CODE_FLOW.md** for implementation details
- Check **PIPELINE_VISUAL.md** for workflow reminders

## 📧 Feedback

If you find the documentation helpful or have suggestions for improvement, please:
1. Star the repository ⭐
2. Cite the paper if you use DINO Teacher in research
3. Open an issue with documentation feedback

---

**Happy Learning! 🎉**

The documentation represents a comprehensive guide to understanding the DINO Teacher codebase, from high-level concepts to implementation details. Start with the visual pipeline guide and dive deeper as needed!
