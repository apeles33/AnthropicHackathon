---
name: ml-research-builder
description: Builds diverse machine learning and deep learning research projects across 15 types including paper recreations, architecture implementations (ResNet, Transformer, U-Net), fine-tuning experiments, training innovations (mixup, curriculum learning, SAM), interpretability studies (attention visualization, Grad-CAM, SHAP), synthetic data generation (GANs, VAEs, diffusion), RL environments, loss function design, model compression (quantization, pruning), few-shot learning, transfer learning, neural architecture search, adversarial ML (FGSM, PGD), multi-modal fusion, and continual learning. Each project produces working PyTorch code, visualizations, and professional HTML presentations. Use when building ML/DL hackathon projects, research demonstrations, or educational examples.
---

# ML Research Builder

Builds complete ML/DL research projects across 15 types. Each includes PyTorch code, visualizations, and HTML presentations.

## 15 Project Types

1. **Paper Recreation** - Reproduce arXiv experiments
2. **Architecture from Scratch** - ResNet, Transformer, U-Net
3. **Fine-Tuning** - BERT, GPT, ViT transfer learning
4. **Training Innovation** - Mixup, curriculum, SAM
5. **Interpretability** - Attention, Grad-CAM, SHAP
6. **Synthetic Data** - GANs, VAEs, diffusion
7. **RL Environment** - CartPole, custom envs
8. **Loss Design** - Focal, triplet, contrastive
9. **Model Compression** - Quantization, pruning
10. **Few-Shot Learning** - Prototypical nets, MAML
11. **Transfer Learning** - Domain adaptation
12. **Neural Architecture Search** - Auto model design
13. **Adversarial ML** - FGSM, PGD, robustness
14. **Multi-Modal** - Text+image, audio+visual
15. **Continual Learning** - No catastrophic forgetting

## Workflow

### 1. Initialize
- Select type (1-15)
- Choose model, dataset, parameters creatively
- Document config

### 2. Research
- For papers: Search arXiv, extract methodology
- Review `references/ml_patterns.md` for templates
- Plan 3-4 visualizations

### 3. Implement
Use `scripts/data_loader.py`:
```python
from data_loader import load_mnist, load_cifar10
train, test = load_cifar10()
```

Follow PyTorch patterns from `references/ml_patterns.md`:
- Imports & setup
- Data preparation
- Model definition
- Training loop
- Evaluation
- Visualization
- Export metrics.json

### 4. Visualize
Use `scripts/viz_utils.py`:
```python
from viz_utils import plot_training_curves, plot_confusion_matrix
plot_training_curves(history, 'results/training.png')
```

Create type-specific plots (see references for details)

### 5. HTML Presentation
Populate `assets/presentation_template.html` with:
- Header, executive summary
- Background, methodology
- Experimental setup
- Results with embedded images (base64)
- Code with syntax highlighting
- Conclusions

### 6. Output
```
/mnt/user-data/outputs/ml-[type]-[slug]/
├── index.html
├── code/
│   ├── experiment.py
│   └── requirements.txt
├── results/
│   ├── figure_*.png
│   ├── metrics.json
│   └── model.pth
└── README.md
```

## Creative Freedom

**Agent chooses:**
- Specific architectures
- Dataset variations
- Hyperparameters
- Baseline comparisons
- Visualization styles

**Agent must:**
- Follow 6-step workflow
- Use provided utilities
- Run in < 20 minutes
- Create 3-4 visualizations
- Generate complete HTML

## Resources

**scripts/**
- `data_loader.py` - Load MNIST, CIFAR-10, Fashion-MNIST, text
- `viz_utils.py` - Training curves, confusion matrix, attention, embeddings, ROC

**references/**
- `project_ideas.md` - 150+ ML project ideas
- `ml_patterns.md` - PyTorch templates, training loops, architectures

**assets/**
- `presentation_template.html` - Professional HTML template

## Quality Checklist
- [ ] Code executes (<20 min)
- [ ] 3-4 visualizations
- [ ] Complete HTML
- [ ] Reproducible
- [ ] Professional quality

See full workflow and type-specific guidelines in references.
