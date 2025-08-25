# File: run_imagenet_continual.py
# Evaluates the continual learning performance (BWT, Avg. Acc) of the CALM 
# framework on the large-scale ImageNet dataset, mirroring the Step 2 experiment.


# Example with 10 tasks (100 classes per task) and 5-shot memory
# python run_imagenet_continual.py --imagenet_root D:\Datasets\imagenet_torchvision --num_tasks 10 --k_shot 5

# A faster, less granular experiment with 5 tasks (500 classes per task) and 1-shot memory
# python run_imagenet_continual.py --imagenet_root D:\Datasets\imagenet_torchvision --num_tasks 2 --k_shot 1

# All at once 1000 class test
# python run_imagenet_continual.py --imagenet_root D:\Datasets\imagenet_torchvision --num_tasks 1 --k_shot 5

#Added transforms.ToTensor() to the pipeline to make it compatible with the DataLoader.

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import Subset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse
from tqdm import tqdm
import ssl
from collections import defaultdict
import os

from memory import EpisodicMemory

# --- FIX for SSL Certificate Error ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- 1. Data Handler for Sequential ImageNet ---

class ImageNetContinualManager:
    """
    A manager to handle the ImageNet dataset for continual learning.
    It splits the 1000 classes into a sequence of tasks.
    """
    def __init__(self, root, split, transform, num_tasks):
        print(f"Loading ImageNet '{split}' split for continual learning...")
        self.dataset = ImageNet(root=root, split=split, transform=transform)
        self.num_tasks = num_tasks
        
        if 1000 % num_tasks != 0:
            raise ValueError(f"1000 classes must be divisible by num_tasks ({num_tasks}).")
        
        self.classes_per_task = 1000 // num_tasks
        
        print("Creating robust class index by verifying files...")
        self.class_index = defaultdict(list)
        for master_idx, (path, class_idx) in enumerate(tqdm(self.dataset.samples, desc="Verifying samples")):
            if os.path.exists(path):
                self.class_index[class_idx].append(master_idx)
        
        total_samples = sum(len(v) for v in self.class_index.values())
        print(f"Index complete. Found {len(self.class_index)} classes and {total_samples} actual samples.")

    def get_class_range_for_task(self, task_id):
        start_class = task_id * self.classes_per_task
        end_class = (task_id + 1) * self.classes_per_task
        return range(start_class, end_class)

    def get_task_datasets(self, task_id):
        """
        Returns a Subset of the data corresponding to a specific task.
        """
        task_indices = []
        class_range = self.get_class_range_for_task(task_id)
        for class_id in class_range:
            task_indices.extend(self.class_index[class_id])
            
        return Subset(self.dataset, task_indices)

# --- 2. Reusable Evaluation Logic ---

def evaluate_on_memory_prototypical(memory, model, processor, test_loader, device, batch_size):
    """Evaluates a test loader using prototypes from the provided memory."""
    model.eval()
    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0: return 0.0
    
    prototypes = prototypes.to(device)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 'images' is a batch of Tensors.
            labels = labels.to(device)
            
            # --- THIS IS THE FIX ---
            # Convert the batch of tensors back to a list of PIL Images to ensure a
            # consistent processing pipeline with how the memory was built.
            pil_images = [transforms.ToPILImage()(img) for img in images]
            
            # Now feed the consistent PIL Image list to the processor.
            image_inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
            query_embeddings = model.get_image_features(**image_inputs)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            
            similarities = query_embeddings @ prototypes.T
            preds_indices = torch.argmax(similarities, dim=1)
            predicted_class_ids = torch.tensor([class_ids[i] for i in preds_indices]).to(device)
            
            correct += (predicted_class_ids == labels).sum().item()
            total += len(labels)
            
    return (correct / total) * 100 if total > 0 else 0.0

def run_continual_task_sequence(model, processor, data_manager, k_shot, device, batch_size):
    """Runs a full continual learning sequence and calculates BWT and Avg. Accuracy."""
    memory = EpisodicMemory()
    num_tasks = data_manager.num_tasks
    results_matrix = np.zeros((num_tasks, num_tasks))

    for task_id in range(num_tasks):
        class_range = data_manager.get_class_range_for_task(task_id)
        print(f"\n--- Learning Task {task_id+1}/{num_tasks} (Classes {class_range.start}-{class_range.stop-1}) ---")
        
        train_subset = data_manager.get_task_datasets(task_id)
        
        for class_id in tqdm(class_range, desc="Building memory for task"):
            class_subset_indices = [i for i in train_subset.indices if train_subset.dataset.samples[i][1] == class_id]
            
            k_sample = min(k_shot, len(class_subset_indices))
            if k_sample < k_shot:
                print(f"Warning: Class {class_id} only has {k_sample} samples in this split.")
            
            support_indices = np.random.choice(class_subset_indices, k_sample, replace=False)
            
            for idx in support_indices:
                # The image here is a tensor, which we convert back to PIL for the processor.
                image_tensor, label = data_manager.dataset[idx]
                image_pil = transforms.ToPILImage()(image_tensor)
                with torch.no_grad():
                    image_input = processor(images=[image_pil], return_tensors="pt").to(device)
                    embedding = model.get_image_features(**image_input).squeeze(0)
                memory.add_example(embedding, label)

        for j in range(task_id + 1):
            test_loader = DataLoader(data_manager.get_task_datasets(j), batch_size=batch_size, shuffle=False)
            acc = evaluate_on_memory_prototypical(memory, model, processor, test_loader, device, batch_size)
            print(f"  Accuracy on Task {j+1} after learning Task {task_id+1}: {acc:.2f}%")
            results_matrix[task_id, j] = acc
    
    avg_accuracy = results_matrix.diagonal().mean()
    bwt = 0.0
    if num_tasks > 1:
        for j in range(num_tasks - 1):
            bwt += results_matrix[num_tasks-1, j] - results_matrix[j, j]
        bwt /= (num_tasks - 1)
        
    return {"avg_acc": avg_accuracy, "bwt": bwt}

# --- 3. Main Orchestrator ---

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Continual Learning on ImageNet")
    parser.add_argument('--imagenet_root', type=str, required=True, help='Root directory of your preprocessed ImageNet.')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of sequential tasks to create from 1000 classes.')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support examples per class.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    return parser.parse_args()

def get_best_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def main():
    args = parse_args()
    device = get_best_device()
    print(f"Using device: {device}")

    print("\n--- WARNING: This is a long-running experiment! ---")
    print(f"It will run a {args.num_tasks}-task continual learning scenario on ImageNet.")
    print("Each task evaluation will become progressively slower.\n")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # --- THIS IS THE FIX ---
    # We add transforms.ToTensor() so the DataLoader can create batches.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # This converts PIL images to Tensors for the DataLoader.
    ])
    
    data_manager = ImageNetContinualManager(root=args.imagenet_root, split='val', transform=transform, num_tasks=args.num_tasks)
    
    print(f"\n--- Running {args.k_shot}-Shot Continual Prototypical Network Sequence ---")
    results = run_continual_task_sequence(model, processor, data_manager, k_shot=args.k_shot, device=device, batch_size=args.batch_size)
    
    print(f"\n\n{'='*60}\n    Final Continual Learning Results on ImageNet\n{'='*60}")
    print(f"{'Metric':<25} | {'Value':<20}")
    print(f"-"*50)
    print(f"{'K-Shot':<25} | {args.k_shot}")
    print(f"{'Number of Tasks':<25} | {args.num_tasks}")
    print(f"-"*50)
    print(f"{'Average Accuracy (%)':<25} | {results['avg_acc']:.2f}")
    print(f"{'Backward Transfer (BWT)':<25} | {results['bwt']:.2f}")
    print(f"-"*50)
    print("\nBWT Definition: A value close to 0 indicates low forgetting. A large negative value indicates severe catastrophic forgetting.")

if __name__ == "__main__":
    main()