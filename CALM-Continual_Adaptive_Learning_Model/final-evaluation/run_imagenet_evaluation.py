# File: run_imagenet_evaluation.py
# The final script to evaluate the few-shot performance of the CALM framework on ImageNet.
#run command: python run_imagenet_evaluation.py --imagenet_root /path/to/your/datasets/imagenet --k_shot 5 --n_way 5

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import argparse
from tqdm import tqdm
import ssl

from memory import EpisodicMemory
from imagenet_sampler import ImageNetSampler

# --- FIX for SSL Certificate Error ---
ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CALM on ImageNet")
    parser.add_argument('--imagenet_root', type=str, required=True, 
                        help='The root directory of your full ImageNet dataset.')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode.')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support examples per class.')
    parser.add_argument('--num_query', type=int, default=15, help='Number of query examples per class.')
    parser.add_argument('--num_episodes', type=int, default=600, 
                        help='Number of episodes to average over for a stable result.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing embeddings.')
    return parser.parse_args()

def get_best_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def run_single_episode(support_set, query_set, model, processor, device, batch_size):
    """Processes a single episode and returns the accuracy."""
    memory = EpisodicMemory()

    # --- 1. Build Memory from Support Set ---
    support_images = [item[0] for item in support_set]
    support_labels = [item[1] for item in support_set]
    
    with torch.no_grad():
        for i in range(0, len(support_images), batch_size):
            batch = support_images[i:i+batch_size]
            image_inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            embeddings = model.get_image_features(**image_inputs)
            # Add each embedding to memory one by one
            for j, embedding in enumerate(embeddings):
                memory.add_example(embedding, support_labels[i+j])

    # --- 2. Evaluate on Query Set ---
    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0: return 0.0
    
    prototypes = prototypes.to(device)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    
    query_images = [item[0] for item in query_set]
    query_labels = torch.tensor([item[1] for item in query_set])
    
    correct_predictions = 0
    with torch.no_grad():
        for i in range(0, len(query_images), batch_size):
            batch = query_images[i:i+batch_size]
            image_inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            query_embeddings = model.get_image_features(**image_inputs)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

            similarities = query_embeddings @ prototypes.T
            pred_indices = torch.argmax(similarities, dim=1)
            
            predicted_class_ids = torch.tensor([class_ids[p] for p in pred_indices]).to(device)
            batch_labels = query_labels[i:i+batch_size].to(device)
            
            correct_predictions += (predicted_class_ids == batch_labels).sum().item()

    return (correct_predictions / len(query_labels)) * 100

def main():
    args = parse_args()
    device = get_best_device()
    print(f"Using device: {device}")

    # --- 1. Load CLIP Model ---
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # --- 2. Load and Prepare ImageNet Data ---
    # CLIP's vision transformer was trained on 224x224 images.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    sampler = ImageNetSampler(root_path=args.imagenet_root, split='val', transform=transform)

    # --- 3. Run Episodic Evaluation ---
    print(f"\nStarting evaluation for {args.num_episodes} episodes...")
    all_accuracies = []
    
    for _ in tqdm(range(args.num_episodes), desc="Running Episodes"):
        support_set, query_set = None, None
        while support_set is None: # Keep trying if a class has too few samples
            support_set, query_set = sampler.sample_episode(args.n_way, args.k_shot, args.num_query)

        accuracy = run_single_episode(support_set, query_set, model, processor, device, args.batch_size)
        all_accuracies.append(accuracy)

    # --- 4. Report Final Results ---
    mean_accuracy = np.mean(all_accuracies)
    std_dev = np.std(all_accuracies)
    # Calculate the 95% confidence interval
    confidence_interval = 1.96 * std_dev / np.sqrt(args.num_episodes)
    
    print("\n--- Final ImageNet Few-Shot Evaluation Results ---")
    print(f"Configuration: {args.n_way}-way, {args.k_shot}-shot with {args.num_query} query images per class.")
    print(f"Evaluated on {args.num_episodes} episodes.")
    print("-" * 50)
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"95% Confidence Interval: +/- {confidence_interval:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()