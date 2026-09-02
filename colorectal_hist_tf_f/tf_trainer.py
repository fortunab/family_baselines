"""
TensorFlow Training Engine & Checkpointing Module for Foundation Models.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers, losses


def compile_and_train_tf_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    output_dir: Path,
    epochs: int = 15,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    active_seed: int = 42
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Learning rate schedule with Cosine Decay
    total_steps = epochs * 50  # approximate steps per epoch
    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps,
        alpha=0.05
    )

    try:
        optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    except AttributeError:
        optimizer = optimizers.Adam(learning_rate=lr_schedule)

    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    ckpt_path = output_dir / "best_tf_model.weights.h5"
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\n[TF-Trainer] Fitting model for {epochs} epochs (Active Seed: {active_seed})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=1
    )

    if ckpt_path.exists():
        print(f"[TF-Trainer] Loading best weights from {ckpt_path}...")
        model.load_weights(str(ckpt_path))

    # Evaluate on the 15% unseen holdout test set
    print("[TF-Trainer] Evaluating on the 15% holdout test dataset...")
    y_test_true, y_test_proba = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_test_proba.append(probs)
        y_test_true.append(labels.numpy())

    y_test_proba = np.vstack(y_test_proba)
    y_test_true = np.concatenate(y_test_true)
    y_test_pred = np.argmax(y_test_proba, axis=1)

    # Plot training curves
    plot_training_curves(history.history, output_dir / "training_curves_tf.png")

    return {
        "y_true_test": y_test_true,
        "y_pred_test": y_test_pred,
        "y_proba_test": y_test_proba,
        "history": history.history
    }


def plot_training_curves(hist: Dict[str, list], output_path: Path):
    epochs = range(1, len(hist.get("loss", [])) + 1)
    plt.figure(figsize=(10, 4), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.subplot(1, 2, 1)
    if "loss" in hist:
        plt.plot(epochs, hist["loss"], label="Train Loss", lw=2)
    if "val_loss" in hist:
        plt.plot(epochs, hist["val_loss"], label="Val Loss", lw=2)
    plt.title("Categorical Cross-Entropy Loss", weight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    if "accuracy" in hist:
        plt.plot(epochs, [a*100 for a in hist["accuracy"]], label="Train Accuracy", lw=2)
    if "val_accuracy" in hist:
        plt.plot(epochs, [a*100 for a in hist["val_accuracy"]], label="Val Accuracy", lw=2)
    plt.title("Classification Accuracy (%)", weight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[TF-Trainer] Training curves saved to: {output_path}")
