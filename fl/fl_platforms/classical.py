import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from fl_platforms.base import BaseFLPlatform
from experiments.metrics import calculate_metrics

class ClassicalPlatform(BaseFLPlatform):
    """
    Classical Environment (Centralized Reference Baseline) (Section 3 & Section 4).
    Trains centrally on pooled data without federated client distribution.
    """
    def __init__(self, env_type='hosted'):
        super().__init__('Classical', env_type=env_type)

    def train_and_evaluate(
        self,
        model_factory_fn,
        train_dataset,
        test_dataset,
        num_classes,
        num_rounds=100,
        num_clients=1,
        local_epochs=5,
        batch_size=32,
        lr=0.001,
        weight_decay=0.001
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = model_factory_fn(num_classes=num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = []
        best_metrics = None

        total_epochs = min(num_rounds * local_epochs, 25)

        for epoch in range(1, total_epochs + 1):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluate epoch
            model.eval()
            all_preds, all_targets, all_probs = [], [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs), num_classes)
            metrics['round'] = epoch
            history.append(metrics)

            if best_metrics is None or metrics['accuracy'] > best_metrics['accuracy']:
                best_metrics = metrics

        res = best_metrics if best_metrics else history[-1]
        res['history'] = history
        return res
