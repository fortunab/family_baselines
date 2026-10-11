from abc import ABC, abstractmethod
import torch

class BaseFLPlatform(ABC):
    """
    Abstract interface for Federated Learning Platforms and Centralized Baseline.
    """
    def __init__(self, platform_name, env_type='hosted'):
        self.platform_name = platform_name
        self.env_type = env_type  # 'hosted' or 'on-premises'

    @abstractmethod
    def train_and_evaluate(
        self,
        model_factory_fn,
        train_dataset,
        test_dataset,
        num_classes,
        num_rounds=100,
        num_clients=4,
        local_epochs=5,
        batch_size=32,
        lr=0.001,
        weight_decay=0.001
    ):
        """
        Executes FL rounds or centralized training and returns evaluation metrics dictionary:
        {
          'accuracy': float,
          'sensitivity': float,
          'specificity': float,
          'roc_auc': float,
          'history': list of dicts per round
        }
        """
        pass
