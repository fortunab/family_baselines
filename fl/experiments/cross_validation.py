import numpy as np
from sklearn.model_selection import KFold
import torch
from torchvision import transforms
from med_datasets.preprocessing import MedicalDataset

def run_10fold_cross_validation(
    platform,
    model_factory_fn,
    dataset_obj,
    num_classes,
    num_rounds=10,
    local_epochs=2,
    batch_size=32,
    lr=0.001,
    weight_decay=0.001
):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    images = dataset_obj.images
    labels = dataset_obj.labels

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        train_imgs, train_lbls = images[train_idx], labels[train_idx]
        val_imgs, val_lbls = images[val_idx], labels[val_idx]

        tf_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        tf_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_ds = MedicalDataset(train_imgs, train_lbls, transform=tf_train)
        val_ds = MedicalDataset(val_imgs, val_lbls, transform=tf_val)

        metrics = platform.train_and_evaluate(
            model_factory_fn,
            train_ds,
            val_ds,
            num_classes=num_classes,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay
        )
        fold_results.append(metrics)

    avg_accuracy = float(np.mean([r['accuracy'] for r in fold_results]))
    avg_sensitivity = float(np.mean([r['sensitivity'] for r in fold_results]))
    avg_specificity = float(np.mean([r['specificity'] for r in fold_results]))
    avg_auc = float(np.mean([r['roc_auc'] for r in fold_results]))

    return {
        'accuracy': round(avg_accuracy, 2),
        'sensitivity': round(avg_sensitivity, 2),
        'specificity': round(avg_specificity, 2),
        'roc_auc': round(avg_auc, 3),
        'folds': fold_results
    }
