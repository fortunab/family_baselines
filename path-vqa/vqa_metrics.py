"""
Standard Clinical VQA Evaluation Metrics & NLP Scorer for PathVQA.
Computes:
1. Overall VQA Exact Match (Accuracy %)
2. Closed-Ended (Yes/No) Accuracy, Precision, Recall, F1
3. Open-Ended BLEU-1, BLEU-4
4. ROUGE-L F1-Score
5. Token-level F1-Score
6. Diagnostic Visual VQA Card
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def compute_token_f1(pred: str, target: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    target_tokens = normalize_answer(target).split()
    if not pred_tokens or not target_tokens:
        return 1.0 if pred_tokens == target_tokens else 0.0

    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return 2.0 * (precision * recall) / (precision + recall)


def compute_bleu(pred: str, target: str, n: int = 1) -> float:
    pred_tokens = normalize_answer(pred).split()
    target_tokens = normalize_answer(target).split()
    if not pred_tokens or not target_tokens:
        return 1.0 if pred_tokens == target_tokens else 0.0

    if len(pred_tokens) < n or len(target_tokens) < n:
        return 1.0 if normalize_answer(pred) == normalize_answer(target) else 0.0

    def get_ngrams(tokens, k):
        return [tuple(tokens[i:i+k]) for i in range(len(tokens)-k+1)]

    pred_ngrams = get_ngrams(pred_tokens, n)
    target_ngrams = get_ngrams(target_tokens, n)

    matches = sum(1 for g in pred_ngrams if g in target_ngrams)
    return float(matches / len(pred_ngrams))


def compute_rouge_l(pred: str, target: str) -> float:
    p_tok = normalize_answer(pred).split()
    t_tok = normalize_answer(target).split()
    if not p_tok or not t_tok:
        return 1.0 if p_tok == t_tok else 0.0

    # Longest common subsequence
    m, n = len(p_tok), len(t_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if p_tok[i] == t_tok[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])

    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2.0 * (prec * rec) / (prec + rec)


def evaluate_vqa_predictions(predictions: List[str], ground_truths: List[str], is_closed_list: List[bool]) -> Dict[str, float]:
    total = len(predictions)
    if total == 0:
        return {}

    exact_matches = []
    token_f1s = []
    bleu_1s = []
    bleu_4s = []
    rouge_ls = []

    closed_preds = []
    closed_targets = []

    for pred, gt, is_closed in zip(predictions, ground_truths, is_closed_list):
        p_norm = normalize_answer(pred)
        g_norm = normalize_answer(gt)

        em = 1.0 if (p_norm == g_norm or g_norm in p_norm) else 0.0
        exact_matches.append(em)

        token_f1s.append(compute_token_f1(pred, gt))
        bleu_1s.append(compute_bleu(pred, gt, n=1))
        bleu_4s.append(compute_bleu(pred, gt, n=4))
        rouge_ls.append(compute_rouge_l(pred, gt))

        if is_closed:
            closed_preds.append(1 if "yes" in p_norm else 0)
            closed_targets.append(1 if "yes" in g_norm else 0)

    # Closed-ended Yes/No metrics
    if len(closed_targets) > 0:
        c_preds = np.array(closed_preds)
        c_targets = np.array(closed_targets)
        tp = np.sum((c_preds == 1) & (c_targets == 1))
        fp = np.sum((c_preds == 1) & (c_targets == 0))
        fn = np.sum((c_preds == 0) & (c_targets == 1))
        tn = np.sum((c_preds == 0) & (c_targets == 0))
        smooth = 1e-6
        c_acc = (tp + tn) / (len(c_targets) + smooth)
        c_prec = tp / (tp + fp + smooth)
        c_rec = tp / (tp + fn + smooth)
        c_f1 = 2.0 * (c_prec * c_rec) / (c_prec + c_rec + smooth)
    else:
        c_acc, c_prec, c_rec, c_f1 = 0.0, 0.0, 0.0, 0.0

    return {
        "Overall_Accuracy_EM": float(np.mean(exact_matches)),
        "Yes_No_Accuracy": float(c_acc),
        "Yes_No_F1": float(c_f1),
        "BLEU_1": float(np.mean(bleu_1s)),
        "BLEU_4": float(np.mean(bleu_4s)),
        "ROUGE_L_F1": float(np.mean(rouge_ls)),
        "Token_F1": float(np.mean(token_f1s)),
        "Total_Samples": int(total),
        "Closed_Samples": int(len(closed_targets))
    }


def print_vqa_report(metrics: Dict[str, Any], model_name: str, seed: Optional[int] = None):
    print("\n" + "="*95)
    print(f"      PATH-VQA FOUNDATION MODEL EVALUATION REPORT: {model_name.upper()}")
    print("="*95)
    if seed is not None:
        print(f" Experiment Seed             : {seed}")
    print(f" Total Evaluated QA Pairs    : {metrics.get('Total_Samples', 'N/A')}")
    print(f" Overall VQA Accuracy (EM)   : {metrics['Overall_Accuracy_EM']*100:.2f}%")
    print(f" Closed-Ended (Yes/No) Acc   : {metrics['Yes_No_Accuracy']*100:.2f}%")
    print(f" Closed-Ended (Yes/No) F1    : {metrics['Yes_No_F1']*100:.2f}%")
    print(f" Linguistic BLEU-1 Score     : {metrics['BLEU_1']*100:.2f}%")
    print(f" Linguistic BLEU-4 Score     : {metrics['BLEU_4']*100:.2f}%")
    print(f" Linguistic ROUGE-L F1-Score : {metrics['ROUGE_L_F1']*100:.2f}%")
    print(f" Semantic Token-level F1     : {metrics['Token_F1']*100:.2f}%")
    print("="*95 + "\n")


def plot_vqa_diagnostic_card(
    image: Image.Image,
    question: str,
    ground_truth: str,
    prediction: str,
    output_path: Path,
    model_name: str = "Foundation Model"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4.5), dpi=300)
    sns.set_theme(style="white")

    # 1. Pathology Micrograph
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("H&E Pathology Micrograph", weight='bold', fontsize=11)
    plt.axis("off")

    # 2. Textual QA Diagnostic Card
    plt.subplot(1, 2, 2)
    plt.axis("off")

    card_text = (
        f"VQA Diagnostic Assessment\n"
        f"Model: {model_name.upper()}\n"
        f"-----------------------------------------\n\n"
        f"Question:\n{question}\n\n"
        f"Ground Truth Answer:\n'{ground_truth}'\n\n"
        f"Model Prediction:\n'{prediction}'\n\n"
        f"Match Status: {'CORRECT' if normalize_answer(ground_truth) in normalize_answer(prediction) else 'DISCREPANCY'}"
    )

    plt.text(
        0.05, 0.50, card_text,
        fontsize=10,
        va='center',
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f5f7fa", edgecolor="#34495e", lw=1.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[VQA-Metrics] Diagnostic card saved to: {output_path}")
