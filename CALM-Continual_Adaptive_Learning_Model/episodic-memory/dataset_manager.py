# File: dataset_manager.py
import torch
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision import transforms
from torch.utils.data import Subset

# --- NEW: Configuration Dictionary ---
# This dictionary holds the default settings for each dataset.
# It makes the manager self-aware and easy to extend.
DATASET_CONFIGS = {
    'cifar100': {
        'default_tasks': 10
    },
    'cifar10': {
        'default_tasks': 5  # A more sensible default for 10 classes (5 tasks of 2 classes each)
    }
}


class DatasetManager:
    """
    A unified interface to handle various datasets for continual learning experiments.
    This class now knows the best default settings for each dataset.
    """
    def __init__(self, dataset_name, num_tasks=None, k_shot=5): # num_tasks default is now None
        """
        Initializes the manager and loads the specified dataset.
        If num_tasks is not provided, it will use the sensible default for that dataset.
        """
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
        
        # --- MODIFIED: Smart Task Handling ---
        # If the user does not specify a number of tasks, use our smart default.
        if num_tasks is None:
            self.num_tasks = DATASET_CONFIGS[self.dataset_name]['default_tasks']
            print(f"No --num_tasks specified. Using smart default of {self.num_tasks} for {self.dataset_name}.")
        else:
            # If the user provides a value, it overrides the default.
            self.num_tasks = num_tasks
            print(f"User override: Using {self.num_tasks} tasks.")
        
        self.k_shot = k_shot
        
        self.train_dataset = None
        self.test_dataset = None
        self.classes_per_task = None
        self.class_names = None
        
        self._load_dataset()
        print(f"DatasetManager initialized for '{self.dataset_name}' with {self.num_tasks} tasks.")
    
    # ... The rest of the file remains exactly the same ...
    def _load_dataset(self):
        """Internal method to load the correct dataset based on the name."""
        transform = transforms.ToTensor()
        
        if self.dataset_name == 'cifar100':
            self.train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
            self.test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)
            total_classes = 100
        elif self.dataset_name == 'cifar10':
            self.train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
            self.test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
            total_classes = 10
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
            
        if total_classes % self.num_tasks != 0:
            raise ValueError(f"Total classes ({total_classes}) must be divisible by num_tasks ({self.num_tasks}).")
        
        self.classes_per_task = total_classes // self.num_tasks
        self.class_names = self.train_dataset.classes

    def get_task_datasets(self, dataset_type, task_id):
        dataset = self.train_dataset if dataset_type == 'train' else self.test_dataset
        start_class = task_id * self.classes_per_task
        end_class = (task_id + 1) * self.classes_per_task
        indices = [i for i, target in enumerate(dataset.targets) if start_class <= target < end_class]
        return Subset(dataset, indices)
        
    def get_class_range_for_task(self, task_id):
        start_class = task_id * self.classes_per_task
        end_class = (task_id + 1) * self.classes_per_task
        return range(start_class, end_class)

    def get_class_names_for_task(self, task_id):
        if not self.class_names:
             raise RuntimeError("Class names have not been loaded. Call _load_dataset first.")
        class_range = self.get_class_range_for_task(task_id)
        return [self.class_names[i] for i in class_range]