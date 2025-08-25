# File: imagenet_sampler.py
# MODIFIED: A simpler, more correct, and robust version.
# It builds the index by verifying that each expected file from the dataset actually exists on disk.

import torch
from torchvision.datasets import ImageNet
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm

class ImageNetSampler:
    def __init__(self, root_path, split, transform):
        """
        Initializes the sampler by loading the ImageNet dataset and creating a
        robust index by verifying the existence of each file on disk.
        """
        print(f"Loading ImageNet '{split}' split from disk...")
        self.dataset = ImageNet(root=root_path, split=split, transform=transform)
        
        print("Creating robust class index by verifying files... (This may take a minute)")
        self.class_index = defaultdict(list)

        # --- THIS IS THE NEW, CORRECTED FIX ---
        # Instead of complex path matching, we iterate through the original list of samples
        # provided by the dataset and only keep the ones that actually exist on disk.
        
        # self.dataset.samples is a list of tuples: (full_path_to_image, class_index)
        for master_idx, (path, class_idx) in enumerate(tqdm(self.dataset.samples, desc="Verifying samples")):
            if os.path.exists(path):
                # If the file exists, it's a valid sample. Add its original index to our list.
                self.class_index[class_idx].append(master_idx)
        
        self.all_class_ids = sorted(self.class_index.keys())
        total_samples = sum(len(v) for v in self.class_index.values())
        print(f"Robust index complete. Found {len(self.all_class_ids)} classes and {total_samples} actual samples.")

        if total_samples == 0:
            raise RuntimeError(
                "The indexer found 0 valid images. This is likely due to a mismatch between the "
                "ImageNet root path and the directory structure. Please ensure the 'val' folder with "
                "class subdirectories is directly inside the provided root path."
            )

    def sample_episode(self, n_way, k_shot, num_query):
        """
        Samples a single N-way, K-shot episode with a query set.
        (This function does not need to be changed).
        """
        episode_classes = np.random.choice(self.all_class_ids, n_way, replace=False)
        
        support_set, query_set = [], []
        
        for class_id in episode_classes:
            all_indices_for_class = self.class_index[class_id]
            
            if len(all_indices_for_class) < k_shot + num_query:
                return None, None
                
            sampled_indices = np.random.choice(
                all_indices_for_class, 
                k_shot + num_query, 
                replace=False
            )
            
            support_indices = sampled_indices[:k_shot]
            query_indices = sampled_indices[k_shot:]
            
            for idx in support_indices:
                support_set.append(self.dataset[idx])
            for idx in query_indices:
                query_set.append(self.dataset[idx])
                
        return support_set, query_set