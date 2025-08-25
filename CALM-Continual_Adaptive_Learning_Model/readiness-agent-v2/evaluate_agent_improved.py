# File: evaluate_agent_improved.py
# Enhanced evaluation script with debugging and multiple calibration methods

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100, STL10, FashionMNIST
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse
from tqdm import tqdm
import ssl

# SSL fix
ssl._create_default_https_context = ssl._create_unverified_context

from memory import EpisodicMemory
from calibration import CalibratedModel  # Uses improved version
from agent import ReadinessAgent  # Uses improved version

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Readiness Agent Evaluation")
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'stl10', 'fashionmnist'],
                        help='Dataset to use for the experiment.')
    parser.add_argument('--k_shot', type=int, default=10, 
                        help='Number of examples per class in memory.')
    parser.add_argument('--threshold', type=float, default=0.7, 
                        help='Initial confidence threshold for the agent.')
    parser.add_argument('--adaptive', action='store_true', default=True,
                        help='Use adaptive threshold calibration.')
    parser.add_argument('--calibration_method', type=str, default='temperature',
                        choices=['temperature', 'isotonic', 'platt', 'none'],
                        help='Confidence calibration method to use.')
    parser.add_argument('--target_deployment', type=float, default=0.75,
                        help='Target deployment rate for adaptive thresholding.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debugging output.')
    return parser.parse_args()

def get_best_device():
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_dataset(name):
    """Enhanced dataset loader with better error handling."""
    print(f"Loading {name.upper()} dataset...")
    
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
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    print(f"Number of classes: {len(train_dataset.classes)}")
    return train_dataset, test_dataset

def build_memory_smart(train_dataset, model, processor, device, k_shot, debug=False):
    """
    Build memory with better class balancing and quality control.
    """
    print(f"\n=== Building {k_shot}-shot episodic memory ===")
    memory = EpisodicMemory()
    
    # Group samples by class
    class_samples = {}
    for idx, (_, label) in enumerate(train_dataset):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(idx)
    
    # Select diverse examples for each class
    for class_idx in range(len(train_dataset.classes)):
        if class_idx not in class_samples:
            print(f"Warning: No samples found for class {class_idx}")
            continue
            
        indices = class_samples[class_idx]
        
        if len(indices) < k_shot:
            print(f"Warning: Class {class_idx} ({train_dataset.classes[class_idx]}) has only {len(indices)} samples")
            selected_indices = indices
        else:
            # Random selection for now - could be improved with diversity-based selection
            selected_indices = np.random.choice(indices, k_shot, replace=False)
        
        # Add selected examples to memory
        class_embeddings = []
        for idx in selected_indices:
            image, label = train_dataset[idx]
            pil_image = transforms.ToPILImage()(image)
            
            with torch.no_grad():
                image_input = processor(images=pil_image, return_tensors="pt").to(device)
                embedding = model.get_image_features(**image_input).squeeze(0)
                memory.add_example(embedding, label)
                class_embeddings.append(embedding)
        
        if debug:
            # Calculate intra-class similarity to check diversity
            if len(class_embeddings) > 1:
                embeddings_tensor = torch.stack(class_embeddings)
                similarity_matrix = torch.cosine_similarity(embeddings_tensor.unsqueeze(1), 
                                                          embeddings_tensor.unsqueeze(0), dim=2)
                avg_similarity = similarity_matrix.sum() / (len(class_embeddings) * (len(class_embeddings) - 1))
                print(f"  Class {class_idx}: {len(selected_indices)} samples, avg similarity: {avg_similarity:.3f}")
    
    print(f"Memory built with {len(memory)} total examples across {len(memory.get_seen_classes())} classes")
    return memory

def run_enhanced_evaluation(args):
    """Main evaluation function with comprehensive analysis."""
    device = get_best_device()
    print(f"Using device: {device}")
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  K-shot: {args.k_shot}")
    print(f"  Initial threshold: {args.threshold:.2%}")
    print(f"  Adaptive: {args.adaptive}")
    print(f"  Calibration: {args.calibration_method}")
    print(f"  Target deployment: {args.target_deployment:.2%}")

    # 1. Load Data and Models
    print("\n" + "="*50)
    print("STEP 1: Loading models and data")
    print("="*50)
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    train_dataset, test_dataset = load_dataset(args.dataset)

    # 2. Create train/validation/test splits
    # Use more validation data for better calibration
    val_size = min(2000, len(test_dataset) // 3 * 2)  # Use 2/3 for validation, but cap at 2000
    test_size = len(test_dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    val_dataset, final_test_dataset = random_split(test_dataset, [val_size, test_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(final_test_dataset, batch_size=32, shuffle=False)
    
    print(f"Data splits: {len(train_dataset)} train, {len(val_dataset)} validation, {len(final_test_dataset)} test")

    # 3. Build Enhanced Memory
    print("\n" + "="*50)
    print("STEP 2: Building episodic memory")
    print("="*50)
    
    memory = build_memory_smart(train_dataset, model, processor, device, args.k_shot, args.debug)

    # 4. Model Calibration with Analysis
    print("\n" + "="*50)
    print("STEP 3: Model calibration and analysis")
    print("="*50)
    
    calibrated_model = CalibratedModel(model, processor, memory, device).to(device)
    
    # Analyze before calibration
    if args.debug:
        calibrated_model.calibration_method = "none"
        raw_conf, _, _ = calibrated_model.analyze_confidence_distribution(val_loader, "Before Calibration")
    
    # Apply calibration
    if args.calibration_method != "none":
        calibrated_model.set_temperature_robust(val_loader, method=args.calibration_method)
        
        # Analyze after calibration
        if args.debug:
            cal_conf, _, _ = calibrated_model.analyze_confidence_distribution(val_loader, "After Calibration")

    # 5. Agent Calibration
    print("\n" + "="*50)
    print("STEP 4: Agent threshold calibration")
    print("="*50)
    
    agent = ReadinessAgent(
        threshold=args.threshold, 
        adaptive=args.adaptive, 
        target_deployment_rate=args.target_deployment
    )
    
    if args.adaptive:
        # Collect validation data for agent calibration
        val_confidences, val_predictions, val_labels = [], [], []
        
        calibrated_model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Collecting validation data"):
                pil_images = [transforms.ToPILImage()(img) for img in images]
                logits = calibrated_model.forward(pil_images)
                confidences = calibrated_model.get_calibrated_confidences(logits)
                _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)
                
                # Map predictions back to original class IDs
                class_ids = sorted(memory.storage.keys())
                mapped_preds = torch.tensor([class_ids[p] for p in predictions], device=device)
                
                val_confidences.extend(confidences.cpu().numpy())
                val_predictions.extend(mapped_preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        # Calibrate agent threshold
        agent.calibrate_threshold(val_confidences, val_predictions, val_labels, verbose=args.debug)

    # 6. Final Evaluation
    print("\n" + "="*50)
    print("STEP 5: Final evaluation on test set")
    print("="*50)
    
    all_confidences, all_preds, all_labels, all_decisions = [], [], [], []

    calibrated_model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final evaluation"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            logits = calibrated_model.forward(pil_images)
            confidences = calibrated_model.get_calibrated_confidences(logits)
            _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)
            
            # Map predictions back to original class IDs
            class_ids = sorted(memory.storage.keys())
            mapped_preds = torch.tensor([class_ids[p] for p in predictions], device=device)

            # Agent decisions
            decisions = [agent.decide(conf.item()) for conf in confidences]
            
            all_confidences.extend(confidences.cpu().numpy())
            all_preds.extend(mapped_preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_decisions.extend(decisions)

    # 7. Results Analysis
    print("\n" + "="*50)
    print(f"FINAL RESULTS: {args.dataset.upper()}")
    print("="*50)
    
    metrics = agent.get_metrics(np.array(all_preds), np.array(all_labels), np.array(all_decisions))
    
    # Main results table
    print(f"\n{'Metric':<35} | {'Value':<12}")
    print("-" * 50)
    print(f"{'Dataset':<35} | {args.dataset.upper():<12}")
    print(f"{'Calibration Method':<35} | {args.calibration_method.capitalize():<12}")
    print(f"{'K-shot Memory':<35} | {args.k_shot:<12}")
    print("-" * 50)
    print(f"{'Base Model Accuracy':<35} | {metrics['Total Model Accuracy']:.2%}")
    print(f"{'Agent Threshold (Calibrated)':<35} | {agent.calibrated_threshold:.3f}")
    print(f"{'Deployment Rate':<35} | {metrics['Total Deployed']:.2%}")
    print(f"{'Feedback Request Rate':<35} | {metrics['Total Requested']:.2%}")
    print("-" * 50)
    print(f"{'Deployment Decision Accuracy':<35} | {metrics['Deployment Decision Accuracy']:.2%}")
    print(f"{'False Positive Rate (FPR)':<35} | {metrics['False Positive Rate (FPR)']:.2%}")
    print(f"{'True Negative Rate (TNR)':<35} | {metrics['True Negative Rate (TNR)']:.2%}")
    print(f"{'Deployed Predictions Accuracy':<35} | {metrics['Deployed Accuracy']:.2%}")
    print("-" * 50)
    
    # Interpretation
    print(f"\n{'INTERPRETATION':<35}")
    print("-" * 50)
    
    if metrics['False Positive Rate (FPR)'] < 0.1:
        fpr_assessment = "EXCELLENT (< 10%)"
    elif metrics['False Positive Rate (FPR)'] < 0.2:
        fpr_assessment = "GOOD (< 20%)"
    elif metrics['False Positive Rate (FPR)'] < 0.3:
        fpr_assessment = "ACCEPTABLE (< 30%)"
    else:
        fpr_assessment = "NEEDS IMPROVEMENT (≥ 30%)"
    
    print(f"{'Safety (FPR)':<35} | {fpr_assessment}")
    
    deployment_rate = metrics['Total Deployed']
    if deployment_rate > 0.8:
        efficiency_assessment = "HIGH EFFICIENCY"
    elif deployment_rate > 0.6:
        efficiency_assessment = "GOOD EFFICIENCY" 
    elif deployment_rate > 0.4:
        efficiency_assessment = "MODERATE EFFICIENCY"
    else:
        efficiency_assessment = "LOW EFFICIENCY"
    
    print(f"{'Efficiency (Deployment Rate)':<35} | {efficiency_assessment}")
    
    # Detailed analysis if requested
    if args.debug:
        agent.detailed_analysis(all_confidences, all_preds, all_labels, all_decisions)
    
    # Recommendations
    print(f"\n{'RECOMMENDATIONS':<35}")
    print("-" * 50)
    
    if metrics['False Positive Rate (FPR)'] > 0.15:
        print("• Consider using a higher threshold or better calibration method")
    
    if metrics['Total Deployed'] < 0.5:
        print("• Threshold may be too conservative - consider lowering it")
        print("• Try improving base model accuracy with more k-shot examples")
    
    if metrics['Deployment Decision Accuracy'] < 0.8:
        print("• Agent decision quality is poor - check confidence calibration")
        print("• Consider using isotonic or Platt scaling instead of temperature scaling")
    
    if metrics['Deployed Accuracy'] < metrics['Total Model Accuracy']:
        print("• Agent may be deploying wrong predictions - investigate threshold calibration")

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        run_enhanced_evaluation(args)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())