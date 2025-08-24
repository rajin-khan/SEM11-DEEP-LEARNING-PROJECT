# File: run_continual_learning.py (Refactored for Comparison)
# run command: python run_continual_learning.py --dataset <dataset_name>

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
import argparse

# Import our modular components
from memory import EpisodicMemory
from dataset_manager import DatasetManager

# --- 1. Setup & Helper Functions ---

def get_best_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Comparison Framework")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10'],
                        help='Dataset to use for the experiment.')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of sequential tasks.')
    return parser.parse_args()

# --- 2. Evaluation Logic Functions ---

def evaluate_zero_shot_continual(model, processor, data_manager, device):
    """Evaluates Zero-Shot performance across all sequential tasks."""
    model.eval()
    task_accuracies = []
    
    # --- FIX 1: Create prompts for ALL classes, not just task-specific ones ---
    all_class_names = data_manager.class_names
    text_prompts = [f"a photo of a {c}" for c in all_class_names]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    for task_id in range(data_manager.num_tasks):
        test_loader = DataLoader(data_manager.get_task_datasets('test', task_id), batch_size=128)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                pil_images = [transforms.ToPILImage()(img) for img in images]
                image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
                
                outputs = model(**image_inputs, **text_inputs)
                logits_per_image = outputs.logits_per_image
                
                # The predictions are now global indices (0-9), matching the original labels
                preds_global = torch.argmax(logits_per_image, dim=1)
                
                # --- FIX 2: Compare directly with original labels. No local conversion needed. ---
                correct += (preds_global == labels.to(device)).sum().item()
                total += len(labels)
        
        task_acc = (correct / total) * 100
        task_accuracies.append(task_acc)
        print(f"  Zero-Shot Accuracy on Task {task_id+1}: {task_acc:.2f}%")
        
    return np.mean(task_accuracies)

def run_continual_task_sequence(model, processor, data_manager, k_shot, device):
    """Runs a full continual learning sequence for a given k_shot value."""
    memory = EpisodicMemory()
    num_tasks = data_manager.num_tasks
    results_matrix = np.zeros((num_tasks, num_tasks))

    for task_id in range(num_tasks):
        class_range = data_manager.get_class_range_for_task(task_id)
        print(f"\n--- Learning Task {task_id+1}/{num_tasks} (Classes {class_range.start}-{class_range.stop-1}) ---")
        
        # a. Learn the new task (Populate memory)
        train_subset = data_manager.get_task_datasets('train', task_id)
        original_targets = np.array(data_manager.train_dataset.targets)[train_subset.indices]
        
        for class_id in class_range:
            class_indices = np.where(original_targets == class_id)[0]
            support_indices = np.random.choice(class_indices, k_shot, replace=False)
            
            for idx in support_indices:
                image, label = train_subset[idx]
                pil_image = transforms.ToPILImage()(image)
                with torch.no_grad():
                    image_input = processor(images=pil_image, return_tensors="pt").to(device)
                    embedding = model.get_image_features(**image_input).squeeze()
                memory.add_example(embedding, label)

        # b. Evaluate on all tasks seen so far
        for j in range(task_id + 1):
            test_loader = DataLoader(data_manager.get_task_datasets('test', j), batch_size=128)
            acc = evaluate_on_memory_prototypical(memory, model, processor, test_loader, device)
            results_matrix[task_id, j] = acc
    
    # c. Calculate final metrics for this run
    avg_accuracy = results_matrix.diagonal().mean()
    bwt = 0.0
    if num_tasks > 1:
        for i in range(1, num_tasks):
            bwt += results_matrix[num_tasks-1, i-1] - results_matrix[i-1, i-1]
        bwt /= (num_tasks - 1)
        
    return {"avg_acc": avg_accuracy, "bwt": bwt}

def evaluate_on_memory_prototypical(memory, model, processor, test_loader, device):
    """Helper function to evaluate using prototypes from memory."""
    model.eval()
    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0: return 0.0
    
    prototypes = prototypes.to(device)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # This line correctly converts the BATCH of images to PIL images
            pil_images = [transforms.ToPILImage()(img) for img in images] 
            
            image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
            query_embeddings = model.get_image_features(**image_inputs)
            
            similarities = query_embeddings @ prototypes.T
            preds_indices = torch.argmax(similarities, dim=1)
            predicted_class_ids = torch.tensor([class_ids[i] for i in preds_indices]).to(device)
            
            correct += (predicted_class_ids == labels.to(device)).sum().item()
            total += len(labels)
            
    return (correct / total) * 100 if total > 0 else 0.0

# --- 3. Main Orchestrator ---

if __name__ == "__main__":
    args = parse_args()
    device = get_best_device()
    
    # Load models and data manager once
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    data_manager = DatasetManager(dataset_name=args.dataset, num_tasks=args.num_tasks, k_shot=0) # k_shot irrelevant here
    
    print(f"\n{'='*60}\nStarting Continual Learning Comparison on {args.dataset.upper()}\n{'='*60}")
    
    # --- Run Zero-Shot Baseline ---
    print("\n--- Phase 1: Evaluating Zero-Shot Baseline ---")
    zero_shot_avg_acc = evaluate_zero_shot_continual(model, processor, data_manager, device)
    
    # --- Run 1-Shot Continual Learning ---
    print("\n\n--- Phase 2: Running 1-Shot Prototypical Network Sequence ---")
    one_shot_results = run_continual_task_sequence(model, processor, data_manager, k_shot=1, device=device)
    
    # --- Run 5-Shot Continual Learning ---
    print("\n\n--- Phase 3: Running 5-Shot Prototypical Network Sequence ---")
    five_shot_results = run_continual_task_sequence(model, processor, data_manager, k_shot=5, device=device)
    
    # --- Final Results Table ---
    print(f"\n\n{'='*60}\n    Final Comparison Results on {args.dataset.upper()}\n{'='*60}")
    print(f"{'Method':<25} | {'Avg. Accuracy (%)':<20} | {'BWT (%)':<15}")
    print(f"-"*60)
    print(f"{'Zero-Shot Baseline':<25} | {zero_shot_avg_acc:<20.2f} | {'N/A':<15}")
    print(f"{'1-Shot Prototypical':<25} | {one_shot_results['avg_acc']:<20.2f} | {one_shot_results['bwt']:<15.2f}")
    print(f"{'5-Shot Prototypical':<25} | {five_shot_results['avg_acc']:<20.2f} | {five_shot_results['bwt']:<15.2f}")
    print(f"-"*60)