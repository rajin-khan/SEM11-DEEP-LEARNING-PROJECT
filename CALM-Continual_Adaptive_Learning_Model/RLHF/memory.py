# File: memory.py
import torch
import torch.nn.functional as F
from collections import defaultdict

class EpisodicMemory:
    """
    A class to manage the episodic memory for a continual learning agent.
    
    This memory stores embeddings and labels from past experiences and can
    compute class prototypes on demand for few-shot classification.
    """
    def __init__(self):
        """Initializes the memory storage."""
        # A dictionary where each key is a class label and each value is a list of
        # embedding tensors for that class.
        self.storage = defaultdict(list)
        print("Episodic Memory initialized.")

    def add_example(self, embedding: torch.Tensor, label: int):
        """
        Adds a single example (embedding and label) to the memory.
        
        Args:
            embedding (torch.Tensor): The embedding vector of the example.
            label (int): The integer label of the example.
        """
        # Ensure the embedding is on the CPU to avoid piling up GPU memory
        # and ensure it's detached from the computation graph.
        self.storage[label].append(embedding.detach().cpu())

    def get_prototypes(self) -> (torch.Tensor, list):
        """
        Calculates the prototype for each class currently in memory.
        
        A prototype is the mean of all embeddings for a class.
        
        Returns:
            A tuple containing:
            - prototypes (torch.Tensor): A tensor of shape [num_classes, embedding_dim]
                                         containing the calculated prototype for each class.
            - class_ids (list): A list of integer labels corresponding to each prototype row.
                                This is crucial because the class IDs might not be sequential
                                (e.g., [0, 1, 5, 8]).
        """
        class_ids = sorted(self.storage.keys())
        if not class_ids:
            return torch.empty(0), []
        
        # Stack all embeddings for each class and calculate the mean
        all_prototypes = [
            torch.stack(self.storage[cid]).mean(dim=0) for cid in class_ids
        ]
        
        prototypes = torch.stack(all_prototypes)
        
        return prototypes, class_ids

    def __len__(self):
        """Returns the total number of examples stored in memory."""
        return sum(len(embeddings) for embeddings in self.storage.values())
        
    def get_seen_classes(self) -> list:
        """Returns a list of unique class labels seen so far."""
        return sorted(self.storage.keys())