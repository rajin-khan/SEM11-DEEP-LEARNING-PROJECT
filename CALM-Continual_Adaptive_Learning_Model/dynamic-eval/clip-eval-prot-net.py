# File: clip-eval-prot-net.py
# Description: Evaluates CLIP using Zero-Shot and a Few-Shot Prototypical Network on a configurable dataset.

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np

# Import shared utilities and configuration
from clip_eval_utils import Config, load_dataset

# --- Evaluation Functions ---

def evaluate_zero_shot(model, processor, test_loader, class_names, device):
    """Evaluates the model using zero-shot classification with text prompts."""
    print("\n--- Starting Zero-Shot Evaluation ---")
    
    text_prompts = [f"a photo of a {c}" for c in class_names]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Zero-Shot"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            image_inputs = processor(images=pil_images, return_tensors="pt").to(device)

            # Get logits by comparing image and text embeddings
            outputs = model(**image_inputs, **text_inputs)
            logits_per_image = outputs.logits_per_image
            
            preds = torch.argmax(logits_per_image, dim=1)
            correct_predictions += (preds == labels.to(device)).sum().item()
            total_samples += len(labels)
    
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

def evaluate_few_shot_prototypical(model, processor, train_dataset, test_loader, class_names, device, k_shot=5):
    """Evaluates using a Prototypical Network on image embeddings."""
    print(f"\n--- Starting {k_shot}-Shot Prototypical Network Evaluation ---")
    num_classes = len(class_names)

    # Step 1: Build the Support Set
    support_images, support_labels = [], []
    train_targets = np.array(train_dataset.targets)
    for class_idx in range(num_classes):
        indices = np.where(train_targets == class_idx)[0]
        k_sample = min(k_shot, len(indices))
        if k_sample < k_shot:
            print(f"Warning: Class {class_names[class_idx]} has only {k_sample} samples, requested {k_shot}.")

        support_indices = np.random.choice(indices, k_sample, replace=False)
        for idx in support_indices:
            image, label = train_dataset[idx]
            support_images.append(image)
            support_labels.append(label)

    # Step 2: Encode the Support Set
    support_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(support_images), Config.BATCH_SIZE), desc=f"Encoding {k_shot}-shot Memory"):
            batch_pil = [transforms.ToPILImage()(img) for img in support_images[i:i+Config.BATCH_SIZE]]
            image_inputs = processor(images=batch_pil, return_tensors="pt").to(device)
            embeddings = model.get_image_features(**image_inputs)
            support_embeddings.append(embeddings)
    support_embeddings = torch.cat(support_embeddings)
    support_labels = torch.tensor(support_labels, device=device)

    # Step 3: Calculate Class Prototypes
    embedding_dim = support_embeddings.size(1)
    prototypes = torch.zeros(num_classes, embedding_dim).to(device)
    for class_idx in range(num_classes):
        class_mask = (support_labels == class_idx)
        if class_mask.sum() == 0: continue
        prototypes[class_idx] = support_embeddings[class_mask].mean(dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    print("Class prototypes created successfully.")

    # Step 4: Evaluate on the Test Set
    correct_predictions, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"{k_shot}-Shot Proto"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
            query_embeddings = model.get_image_features(**image_inputs)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            
            similarities = query_embeddings @ prototypes.T
            preds = torch.argmax(similarities, dim=1)
            
            correct_predictions += (preds == labels.to(device)).sum().item()
            total_samples += len(labels)
            
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Using device: {Config.DEVICE}")
    print(f"Running Prototypical Network evaluation for dataset: {Config.DATASET_NAME}")

    # Load Model and Processor
    model = CLIPModel.from_pretrained(Config.MODEL_NAME).to(Config.DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)

    # Load Data using the utility function
    train_dataset, test_dataset, class_names = load_dataset(Config.DATASET_NAME, Config.DATA_ROOT)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    # --- Run Evaluations for 0, 1, and 5 shots ---
    results = {}

    # Run Zero-Shot
    results["Zero-Shot"] = evaluate_zero_shot(model, processor, test_loader, class_names, Config.DEVICE)

    # Run Few-Shot Prototypical Networks
    k_shots_to_test = [1, 5]
    for k in k_shots_to_test:
        key = f"{k}-Shot ProtoNet"
        results[key] = evaluate_few_shot_prototypical(model, processor, train_dataset, test_loader, class_names, Config.DEVICE, k_shot=k)

    # --- Print Summary ---
    print(f"\n\n--- Evaluation Summary for {Config.DATASET_NAME} (ProtoNet Method) ---")
    for method, acc in results.items():
        print(f"{method:<20}: {acc:.2f}%")