# File: clip.py

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def get_best_device():
    """
    Checks for available hardware and returns the best device for PyTorch.
    Priority: CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        print("CUDA device found. Using NVIDIA GPU.")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("MPS device found. Using Apple Silicon GPU.")
        return "mps"
    else:
        print("No GPU found. Using CPU. Note: This will be significantly slower.")
        return "cpu"

def classify_image(image_path: str, candidate_labels: list):
    """
    Classifies an image against a list of candidate labels using CLIP.
    
    Args:
        image_path (str): The path to the image file.
        candidate_labels (list): A list of strings to classify the image against.
        
    Returns:
        A dictionary containing the predicted label and confidence scores.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None

    # Move the model and inputs to the best available device
    device = get_best_device()
    model.to(device)
    
    # Process the inputs
    inputs = processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Prepare the results
    results = {}
    for label, prob in zip(candidate_labels, probs[0]):
        results[label] = prob.cpu().item() # Move tensor to CPU before getting value

    best_label = max(results, key=results.get)
    
    return {
        "predicted_label": best_label,
        "confidence": results[best_label],
        "all_scores": results
    }


# --- Main execution block ---
if __name__ == "__main__":
    
    # --- 1. Load the Model and Processor ---
    # This happens only once when the script starts.
    print("Loading CLIP model... (This may take a moment on first run)")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("Model loaded successfully.")
    
    # --- 2. Define Inputs ---
    IMAGE_TO_TEST = "dog.jpeg" # IMPORTANT: Make sure this file exists!
    LABELS_TO_TEST = ["a photo of a dog", "a photo of a cat", "a photo of a car", "a picture of a landscape"]
    
    # --- 3. Run Classification and Print Results ---
    classification_result = classify_image(IMAGE_TO_TEST, LABELS_TO_TEST)
    
    if classification_result:
        print("\n--- Results ---")
        print(f"Image: '{IMAGE_TO_TEST}'")
        print(f"Predicted Label: '{classification_result['predicted_label']}'")
        print(f"Confidence Score: {classification_result['confidence'] * 100:.2f}%")
        
        print("\nFull Probability Distribution:")
        for label, score in sorted(classification_result['all_scores'].items(), key=lambda item: item[1], reverse=True):
            print(f"- {label}: {score * 100:.2f}%")