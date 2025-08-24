# File: evaluate_agent.py
# Main script to evaluate the performance of the ReadinessAgent.
# MODIFIED: Now supports multiple datasets (CIFAR-10, CIFAR-100, STL10, FashionMNIST).
# run command: python evaluate_agent.py --dataset <dataset_name> --k_shot 5 --threshold 0.85

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100, STL10, FashionMNIST
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse
from tqdm import tqdm
import ssl

# --- FIX for SSL Certificate Error ---
ssl._create_default_https_context = ssl._create_unverified_context
# ------------------------------------

from memory import EpisodicMemory
from calibration import CalibratedModel
from agent import ReadinessAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Readiness Agent")
    # --- MODIFIED: Added more dataset choices ---
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'stl10', 'fashionmnist'],
                        help='Dataset to use for the experiment.')
    parser.add_argument('--k_shot', type=int, default=20, 
                        help='Number of examples per class in memory. (Increased default for better confidence)')
    parser.add_argument('--threshold', type=float, default=0.90, 
                        help='Confidence threshold for the agent. (Lowered default)')
    return parser.parse_args()

def get_best_device():
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# --- NEW: Flexible dataset loader ---
def load_dataset(name):
    """Loads the specified dataset and returns the train set, test set, and class names."""
    if name == 'cifar10':
        transform = transforms.ToTensor()
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif name == 'cifar100':
        transform = transforms.ToTensor()
        train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif name == 'stl10':
        transform = transforms.ToTensor()
        train_dataset = STL10(root="./data", split='train', download=True, transform=transform)
        test_dataset = STL10(root="./data", split='test', download=True, transform=transform)
        train_dataset.targets = train_dataset.labels
        test_dataset.targets = test_dataset.labels
    elif name == 'fashionmnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        train_dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return train_dataset, test_dataset

def main():
    args = parse_args()
    device = get_best_device()
    print(f"Using device: {device}")
    print(f"Running experiment on: {args.dataset.upper()} with k_shot={args.k_shot} and threshold={args.threshold:.2%}")

    # 1. Load Data and Models
    print("\n--- 1. Loading models and data ---")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    train_dataset, test_dataset = load_dataset(args.dataset)

    # 2. Split Test Set into Validation and Final Test
    val_size = len(test_dataset) // 2
    test_size = len(test_dataset) - val_size
    
    # Ensure reproducibility of the split
    generator = torch.Generator().manual_seed(42)
    val_dataset, final_test_dataset = random_split(test_dataset, [val_size, test_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(final_test_dataset, batch_size=64)
    print(f"Data ready: {len(train_dataset)} train, {len(val_dataset)} validation, {len(final_test_dataset)} test samples.")

    # 3. Build Episodic Memory
    print(f"\n--- 2. Building {args.k_shot}-shot episodic memory ---")
    memory = EpisodicMemory()
    for class_idx in range(len(train_dataset.classes)):
        indices = np.where(np.array(train_dataset.targets) == class_idx)[0]
        if len(indices) < args.k_shot:
            print(f"Warning: Class {class_idx} has only {len(indices)} samples, using all for memory.")
            support_indices = indices
        else:
            support_indices = np.random.choice(indices, args.k_shot, replace=False)
            
        for idx in support_indices:
            image, label = train_dataset[idx]
            pil_image = transforms.ToPILImage()(image)
            with torch.no_grad():
                image_input = processor(images=pil_image, return_tensors="pt").to(device)
                embedding = model.get_image_features(**image_input).squeeze(0)
            memory.add_example(embedding, label)

    # 4. Calibrate the Model
    print("\n--- 3. Calibrating model confidence ---")
    calibrated_model = CalibratedModel(model, processor, memory, device).to(device)
    calibrated_model.set_temperature(val_loader)

    # 5. Evaluate the Readiness Agent
    print("\n--- 4. Evaluating Readiness Agent on the final test set ---")
    agent = ReadinessAgent(threshold=args.threshold)
    
    all_preds, all_labels, all_decisions = [], [], []

    calibrated_model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Agent Evaluation"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            logits = calibrated_model.forward(pil_images)
            scaled_logits = calibrated_model.temperature_scale(logits)
            
            probabilities = torch.softmax(scaled_logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            
            class_ids = sorted(memory.storage.keys())
            mapped_preds = torch.tensor([class_ids[p] for p in predictions], device=device)

            for conf in confidences:
                all_decisions.append(agent.decide(conf.item()))
            
            all_preds.extend(mapped_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Report Results
    print(f"\n--- 5. Final Results for {args.dataset.upper()} ---")
    metrics = agent.get_metrics(np.array(all_preds), np.array(all_labels), np.array(all_decisions))

    print(f"{'Metric':<30} | {'Value':<10}")
    print("-" * 43)
    print(f"{'Agent Confidence Threshold':<30} | {args.threshold:.2%}")
    print(f"{'Overall Model Accuracy':<30} | {metrics['Total Model Accuracy']:.2%}")
    print(f"{'Percentage Deployed':<30} | {metrics['Total Deployed']:.2%}")
    print("-" * 43)
    print(f"{'Deployment Decision Accuracy':<30} | {metrics['Deployment Decision Accuracy']:.2%}")
    print(f"{'False Positive Rate (FPR)':<30} | {metrics['False Positive Rate (FPR)']:.2%}")
    print("-" * 43)
    print("\nFPR Definition: Of all the times the model was WRONG, the percentage the agent still deployed.")
    print("A low FPR is critical for a reliable system.")

if __name__ == "__main__":
    main()