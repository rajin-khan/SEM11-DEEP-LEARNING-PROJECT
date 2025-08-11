# File: dataset_manager.py
import torch
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision import transforms
from torch.utils.data import Subset

class DatasetManager:
    """
    A unified interface to handle various datasets for continual learning experiments.
    
    This class abstracts away the details of loading and splitting datasets,
    allowing the main training loop to be dataset-agnostic.
    """
    def __init__(self, dataset_name, num_tasks, k_shot):
        """
        Initializes the manager and loads the specified dataset.
        
        Args:
            dataset_name (str): The name of the dataset to use (e.g., 'cifar100', 'cifar10').
            num_tasks (int): The number of sequential tasks to split the dataset into.
            k_shot (int): The number of examples per class to use for the support set.
                         (Note: This argument is not used by the manager itself but is
                          kept for a consistent interface with the main script's args).
        """
        self.dataset_name = dataset_name.lower()
        self.num_tasks = num_tasks
        self.k_shot = k_shot # Not used here, but good practice to keep track of it
        
        self.train_dataset = None
        self.test_dataset = None
        self.classes_per_task = None
        self.class_names = None
        
        self._load_dataset()
        print(f"DatasetManager initialized for '{self.dataset_name}' with {self.num_tasks} tasks.")

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
        """
        Gets a subset of the data corresponding to a specific task.
        
        Args:
            dataset_type (str): 'train' or 'test'.
            task_id (int): The ID of the task (from 0 to num_tasks-1).
            
        Returns:
            A torch.utils.data.Subset object.
        """
        dataset = self.train_dataset if dataset_type == 'train' else self.test_dataset
        
        start_class = task_id * self.classes_per_task
        end_class = (task_id + 1) * self.classes_per_task
        
        # This works because dataset.targets is a standard attribute in torchvision datasets
        indices = [i for i, target in enumerate(dataset.targets) if start_class <= target < end_class]
        return Subset(dataset, indices)
        
    def get_class_range_for_task(self, task_id):
        """Returns the start and end class ID for a given task."""
        start_class = task_id * self.classes_per_task
        end_class = (task_id + 1) * self.classes_per_task
        return range(start_class, end_class)

    def get_class_names_for_task(self, task_id):
        """
        Returns the list of class name strings for a given task.
        
        Args:
            task_id (int): The ID of the task (from 0 to num_tasks-1).
        
        Returns:
            A list of strings with the class names for that task.
        """
        if not self.class_names:
             raise RuntimeError("Class names have not been loaded. Call _load_dataset first.")

        class_range = self.get_class_range_for_task(task_id)
        return [self.class_names[i] for i in class_range]