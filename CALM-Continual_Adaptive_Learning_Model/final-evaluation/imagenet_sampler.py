# File: imagenet_sampler.py
# Handles loading ImageNet and sampling N-way, K-shot episodes.

import torch
from torchvision.datasets import ImageNet
from collections import defaultdict
import numpy as np

class ImageNetSampler:
    def __init__(self, root_path, split, transform):
        """
        Initializes the sampler by loading the ImageNet dataset and creating an
        index of samples for each class.
        
        Args:
            root_path (str): The root directory of the ImageNet dataset.
            split (str): 'train' or 'val'. For few-shot, 'val' is common.
            transform: The torchvision transforms to apply to the images.
        """
        print(f"Loading ImageNet '{split}' split... (This might take a moment)")
        # This class from torchvision handles the complex directory structure
        self.dataset = ImageNet(root=root_path, split=split, transform=transform)
        
        print("Creating class index for fast sampling...")
        # This index is the key to fast episode creation.
        # It's a dictionary mapping: {class_id: [list of sample indices]}
        self.class_index = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset.samples):
            self.class_index[label].append(idx)
            
        self.all_class_ids = sorted(self.class_index.keys())
        print(f"Found {len(self.all_class_ids)} classes and {len(self.dataset)} total samples.")

    def sample_episode(self, n_way, k_shot, num_query):
        """
        Samples a single N-way, K-shot episode with a query set.
        
        Returns:
            A tuple of (support_set, query_set), where each is a list of
            (image_tensor, label) tuples.
        """
        # 1. Randomly select N classes for this episode
        episode_classes = np.random.choice(self.all_class_ids, n_way, replace=False)
        
        support_set, query_set = [], []
        
        for class_id in episode_classes:
            all_indices_for_class = self.class_index[class_id]
            
            # 2. Sample K+Q unique indices for this class
            # Ensure we have enough samples for the class
            if len(all_indices_for_class) < k_shot + num_query:
                # This is rare on validation set but good practice
                # print(f"Warning: Class {class_id} has only {len(all_indices_for_class)} samples. Skipping for this episode.")
                return None, None # Return None to signal the caller to resample
                
            sampled_indices = np.random.choice(
                all_indices_for_class, 
                k_shot + num_query, 
                replace=False
            )
            
            # 3. Split into support and query sets
            support_indices = sampled_indices[:k_shot]
            query_indices = sampled_indices[k_shot:]
            
            for idx in support_indices:
                support_set.append(self.dataset[idx])
            for idx in query_indices:
                query_set.append(self.dataset[idx])
                
        return support_set, query_set