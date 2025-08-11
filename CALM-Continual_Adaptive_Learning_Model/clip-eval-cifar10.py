# File: clip-eval-cifar10.py
# script of first updates

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import numpy as np

# --- 1. Setup: Model, Device, and Data ---

def get_best_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

device = get_best_device()
print(f"Using device: {device}")

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval() # Set model to evaluation mode

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

# CIFAR-10 class names
class_names = train_dataset.classes
print(f"CIFAR-10 classes: {class_names}")

# --- 2. Zero-Shot Evaluation Function (Unchanged) ---

def evaluate_zero_shot(test_loader):
    print("\n--- Starting Zero-Shot Evaluation ---")
    
    text_prompts = [f"a photo of a {c}" for c in class_names]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Zero-Shot"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            image_inputs = processor(images=pil_images, return_tensors="pt").to(device)

            outputs = model(**image_inputs, **text_inputs)
            logits = outputs.logits_per_image
            
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels.to(device)).sum().item()
            total_samples += len(labels)
    
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

# --- 3. Few-Shot K-NN Evaluation Function (MODIFIED TO BE FLEXIBLE) ---

def evaluate_few_shot_knn(train_dataset, test_loader, k_shot=1):
    """
    Evaluates the model using a K-Nearest Neighbors approach.
    This function is now generalized to handle any number of shots (k_shot).
    """
    print(f"\n--- Starting {k_shot}-Shot K-NN Evaluation ---")

    # --- Step 3a: Build the Support Set (Episodic Memory) ---
    # Get k_shot examples for each class.
    support_images = []
    support_labels = []
    
    for class_idx, class_name in enumerate(tqdm(class_names, desc=f"Building {k_shot}-shot Memory")):
        # Find all indices for the current class in the training dataset
        indices = np.where(np.array(train_dataset.targets) == class_idx)[0]
        # Randomly sample k_shot indices. This is more robust than taking the first k.
        support_indices = np.random.choice(indices, k_shot, replace=False)
        
        for idx in support_indices:
            image, label = train_dataset[idx]
            support_images.append(image)
            support_labels.append(label)

    # --- Step 3b: Encode the Support Set to get embeddings ---
    print("Encoding images in memory...")
    support_embeddings = []
    with torch.no_grad():
        # Process in batches for efficiency
        batch_size = 32
        for i in range(0, len(support_images), batch_size):
            batch_pil = [transforms.ToPILImage()(img) for img in support_images[i:i+batch_size]]
            image_inputs = processor(images=batch_pil, return_tensors="pt").to(device)
            embeddings = model.get_image_features(**image_inputs).cpu().numpy()
            support_embeddings.append(embeddings)

    support_embeddings = np.concatenate(support_embeddings)

    # --- Step 3c: Train a K-Nearest Neighbors classifier ---
    # The number of neighbors 'k' is set to be the same as the number of shots.
    # For 5-shot, this means we find the 5 nearest neighbors in our memory and vote on the class.
    knn = KNeighborsClassifier(n_neighbors=k_shot)
    knn.fit(support_embeddings, support_labels)
    print(f"K-NN classifier trained with k={k_shot}.")


    # --- Step 3d: Evaluate on the test set ---
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"{k_shot}-Shot K-NN"):
            pil_images = [transforms.ToPILImage()(img) for img in images]
            image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
            query_embeddings = model.get_image_features(**image_inputs).cpu().numpy()
            
            preds = knn.predict(query_embeddings)
            correct_predictions += (preds == labels.numpy()).sum()
            total_samples += len(labels)
            
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy

# --- 4. Main Execution ---

if __name__ == "__main__":
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Run and print Zero-Shot results
    zero_shot_accuracy = evaluate_zero_shot(test_loader)
    
    # Run and print 1-Shot K-NN results
    one_shot_accuracy = evaluate_few_shot_knn(train_dataset, test_loader, k_shot=1)
    
    # Run and print 5-Shot K-NN results
    five_shot_accuracy = evaluate_few_shot_knn(train_dataset, test_loader, k_shot=5)

    print("\n--- K-NN Method Summary ---")
    print(f"Final Zero-Shot Accuracy on CIFAR-10: {zero_shot_accuracy:.2f}%")
    print(f"Final 1-Shot K-NN Accuracy on CIFAR-10: {one_shot_accuracy:.2f}%")
    print(f"Final 5-Shot K-NN Accuracy on CIFAR-10: {five_shot_accuracy:.2f}%")