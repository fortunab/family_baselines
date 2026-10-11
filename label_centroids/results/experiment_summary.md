# Experimental Results: LACNet Histology Classification
*Paper: "Label-Attractor Contrastive Network: A Framework for Centroid-Driven Histology Samples"*

## Table 1. Comparison with State-of-the-Art Methods
| Method | Accuracy (%) | Macro F1 |
| :--- | :---: | :---: |
| Standard CNN (Softmax) | 77.5 | 0.78 |
| Metric learning (Triplet loss) | 75.6 | 0.73 |
| Prototypical network | 77.2 | 0.75 |
| Supervised contrastive CNN | 79.1 | 0.78 |
| Multi-prototype CNN | 80.3 | 0.79 |
| **LACNet (our)** | **84.1** | **0.84** |

## Table 2. Class-Wise Performance of LACNet
| Class | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: |
| Class 0: tumour epithelium | 0.92 | 0.90 | 0.91 |
| Class 1: simple stroma | 0.67 | 0.70 | 0.68 |
| Class 2: complex stroma | 0.77 | 0.73 | 0.75 |
| Class 3: immune cell conglomerates | 0.97 | 0.89 | 0.93 |
| Class 4: debris and mucus | 0.83 | 0.80 | 0.81 |
| Class 5: mucosal glands | 0.79 | 0.90 | 0.84 |
| Class 6: adipose tissue | 0.98 | 0.82 | 0.90 |
| Class 7: background | 0.85 | 0.98 | 0.91 |
| **Macro avg.** | **0.85** | **0.84** | **0.84** |

## Table 3. Ablation Study of LACNet Components
| Configuration | Accuracy (%) |
| :--- | :---: |
| **Full LACNet** | **84.1** |
| Without centroid refinement | 76.3 |
| Static (non-trainable) centroids | 71.5 |
| Without attractor regularization | 78.3 |
| Softmax classifier instead of attractors | 77.5 |

## Visualizations
- Figure 3 (CNN Embeddings t-SNE): `results/fig3_cnn_tsne.png`
- Figure 4 (LACNet Embeddings t-SNE): `results/fig4_lacnet_tsne.png`
