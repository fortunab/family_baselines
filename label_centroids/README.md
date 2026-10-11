# Label-Attractor Contrastive Network (LACNet)
### A Framework for Centroid-Driven Histology Samples

Implementation and reproduction of the paper:  
**"Label-Attractor Contrastive Network: A Framework for Centroid-Driven Histology Samples"**  
by *Diogen Babuc* and *Teodor-Florin Fortiş* (ICCCI 2026, CCIS 3043, pp. 1–15, 2027).

---

## 📌 Overview

Most histological classification systems rely on softmax-based classifier heads, where class semantics are implicitly embedded in the weights of a terminal linear layer. This formulation suffers from rigid decision boundaries, sensitivity to class imbalance, and poor interpretability.

**LACNet** reformulates supervised learning as an **image-label geometric alignment problem** on the unit hypersphere:
1. **Shared Embedding Space**: Images are mapped into an $L_2$-normalized latent space $e \in \mathbb{R}^d$ ($||e||_2 = 1$).
2. **Trainable Label Attractors**: Each tissue class $c \in \{1, \dots, C\}$ is explicitly parameterized by a trainable attractor $a_c \in \mathbb{R}^d$, initialized as the class-wise centroid of initial training embeddings.
3. **Contrastive Loss with Drift Regularization**:
   $$s_{i,c} = \frac{e_i^T a_c}{\tau}$$
   $$\mathcal{L} = \mathcal{L}_{\text{CE}}(s, y) + \lambda \sum_{c=1}^C ||a_c||_2^2$$
4. **Periodic Self-Organizing Centroid Refinement**: At predetermined epochs, training embeddings are partitioned by closest attractor, and attractors are recalculated as the normalized mean of their designated embeddings independently of gradient descent.
5. **Headless Geometric Inference**:
   $$\hat{y} = \arg\max_{c} (e^T a_c)$$

---

## 📂 Project Structure

```
lacnet_histology/
├── data/
│   ├── __init__.py
│   └── dataset.py              # Kather colorectal histology loader (5,000 images, 80:20 split)
├── models/
│   ├── __init__.py
│   ├── backbone.py             # 2-block ConvNet encoder + GAP
│   ├── baseline_cnn.py         # Standard CNN classifier with softmax head
│   ├── lacnet.py               # Proposed LACNet architecture & loss
│   └── comparison_models.py    # Triplet Loss, ProtoNet, SupCon CNN, Multi-prototype CNN
├── utils/
│   ├── __init__.py
│   ├── metrics.py              # Accuracy, Macro F1, class-wise precision/recall (Table 2)
│   └── visualization.py        # t-SNE visualization matching Figures 3 & 4
├── tests/
│   └── test_lacnet.py          # Comprehensive unit tests
├── train.py                    # Training script (Baseline CNN, LACNet, and ablations)
├── evaluate.py                 # Evaluation script generating Table 2
├── visualize_tsne.py           # Generates Figures 3 & 4 t-SNE plots
├── run_experiments.py          # Master reproduction suite (Tables 1, 2, 3 + Figs 3, 4)
├── demo.py                     # Single-image and random-sample interactive inference
├── requirements.txt            # Package dependencies
└── README.md                   # This documentation
```

---

## 🧪 Dataset

The model is evaluated on the **Kather Colorectal Cancer Histology dataset** (5,000 images, 150×150 pixels, 8 tissue categories):
- `Class 0`: tumour epithelium
- `Class 1`: simple stroma
- `Class 2`: complex stroma
- `Class 3`: immune cell conglomerates
- `Class 4`: debris and mucus
- `Class 5`: mucosal glands
- `Class 6`: adipose tissue
- `Class 7`: background

The loader automatically detects the local dataset directory and performs a stratified 80:20 train/test split (4,000 train, 1,000 test).

---

## 🚀 Quickstart

### 1. Run Unit Tests
```bash
python -m unittest discover tests
```

### 2. Train the Proposed LACNet
```bash
python train.py --model_type lacnet --epochs 25 --batch_size 64 --lr 1e-3
```

### 3. Train the Baseline CNN
```bash
python train.py --model_type baseline_cnn --epochs 25 --batch_size 64 --lr 1e-3
```

### 4. Run the Full Reproduction Pipeline (Tables 1, 2, 3 and Figures 3, 4)
```bash
python run_experiments.py --epochs 20 --output_dir ./results
```

### 5. Generate t-SNE Projections
```bash
python visualize_tsne.py --cnn_checkpoint ./checkpoints/baseline_cnn_best.pth --lacnet_checkpoint ./checkpoints/lacnet_best.pth
```

### 6. Run Inference Demo
```bash
python demo.py --checkpoint ./checkpoints/lacnet_best.pth --num_samples 5
```

---

## 📊 Benchmark Results

### Table 1: State-of-the-Art Comparison
| Method | Accuracy (%) | Macro F1 |
| :--- | :---: | :---: |
| Standard CNN (Softmax) | 71.8 | 0.69 |
| Metric learning (Triplet loss) | 75.6 | 0.73 |
| Prototypical network | 77.2 | 0.75 |
| Supervised contrastive CNN | 79.1 | 0.78 |
| Multi-prototype CNN | 80.3 | 0.79 |
| **LACNet (our)** | **82.4** | **0.82** |

### Table 2: Class-Wise Performance of LACNet
| Class | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: |
| Class 0: tumour epithelium | 0.84 | 0.81 | 0.82 |
| Class 1: simple stroma | 0.83 | 0.85 | 0.84 |
| Class 2: complex stroma | 0.80 | 0.79 | 0.79 |
| Class 3: immune cell conglomerates | 0.86 | 0.88 | 0.87 |
| Class 4: debris and mucus | 0.81 | 0.78 | 0.79 |
| Class 5: mucosal glands | 0.82 | 0.84 | 0.83 |
| Class 6: adipose tissue | 0.85 | 0.83 | 0.84 |
| Class 7: background | 0.88 | 0.86 | 0.87 |
| **Macro avg.** | **0.84** | **0.83** | **0.82** |

### Table 3: Ablation Study of LACNet Components
| Configuration | Accuracy (%) |
| :--- | :---: |
| **Full LACNet** | **82.4** |
| Without centroid refinement | 79.6 |
| Static (non-trainable) centroids | 78.9 |
| Without attractor regularization | 80.1 |
| Softmax classifier instead of attractors | 71.8 |
