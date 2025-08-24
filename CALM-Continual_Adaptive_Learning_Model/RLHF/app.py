# File: app.py
# A dataset-agnostic Streamlit app to demonstrate the human-in-the-loop feedback for CALM.
# Supports CIFAR-10, CIFAR-100, STL10, and FashionMNIST.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, STL10, FashionMNIST
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from collections import Counter

# Import our memory class
from memory import EpisodicMemory

# --- 1. Model and Data Loading (Cached for Performance) ---

@st.cache_resource
def load_clip_model():
    """Loads the CLIP model and processor. Cached so it only runs once."""
    st.write("Loading CLIP model... (this will only happen once)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

@st.cache_data
def load_dataset(name):
    """Loads the specified dataset and returns the train set and class names."""
    if name == 'CIFAR-10':
        transform = transforms.ToTensor()
        train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        class_names = train_dataset.classes
    elif name == 'CIFAR-100':
        transform = transforms.ToTensor()
        train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
        class_names = train_dataset.classes
    elif name == 'STL10':
        transform = transforms.ToTensor()
        train_dataset = STL10(root="./data", split='train', download=True, transform=transform)
        # STL10 uses '.labels' instead of '.targets'
        train_dataset.targets = train_dataset.labels
        class_names = train_dataset.classes
    elif name == 'FashionMNIST':
        # CLIP requires RGB images. We need to convert grayscale to 3-channel.
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        train_dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
        class_names = train_dataset.classes
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return train_dataset, class_names

def initialize_memory(k_shot, dataset, model, processor, device):
    """Creates a new EpisodicMemory instance and populates it with k-shot examples."""
    memory = EpisodicMemory()
    class_names = dataset.classes
    
    progress_bar = st.progress(0, text="Building memory...")
    
    for i, class_name in enumerate(class_names):
        # Find all indices for the current class
        indices = np.where(np.array(dataset.targets) == i)[0]
        if len(indices) < k_shot:
            st.warning(f"Class '{class_name}' has only {len(indices)} samples, but {k_shot} were requested. Using all available.")
            k_sample = len(indices)
        else:
            k_sample = k_shot
        
        support_indices = np.random.choice(indices, k_sample, replace=False)
        
        for idx in support_indices:
            image, label = dataset[idx]
            pil_image = transforms.ToPILImage()(image)
            with torch.no_grad():
                image_input = processor(images=pil_image, return_tensors="pt").to(device)
                embedding = model.get_image_features(**image_input).squeeze(0)
            memory.add_example(embedding, label)
        
        progress_bar.progress((i + 1) / len(class_names), text=f"Processing class '{class_name}'")
        
    progress_bar.empty()
    return memory

# --- 2. Core Classification Logic ---

def classify_image(image, model, processor, memory, class_names, device):
    """Classifies an image using the prototypes from the episodic memory."""
    with torch.no_grad():
        image_input = processor(images=image, return_tensors="pt").to(device)
        query_embedding = model.get_image_features(**image_input)
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

    prototypes, class_ids = memory.get_prototypes()
    if prototypes.nelement() == 0:
        return "Memory is empty", None, 0.0

    prototypes = prototypes.to(device)
    prototypes = F.normalize(prototypes, p=2, dim=1)

    similarities = query_embedding @ prototypes.T
    confidence, pred_index = similarities.max(dim=1)
    
    predicted_class_id = class_ids[pred_index.item()]
    predicted_class_name = class_names[predicted_class_id]
    
    return predicted_class_name, predicted_class_id, confidence.item()

# --- 3. Streamlit App UI and State Management ---

st.set_page_config(layout="wide")
st.title("🧠 CALM: Human Feedback Loop")
st.write("A demonstration of how CALM learns from human corrections in real-time.")

# --- Sidebar Controls ---
st.sidebar.header("Experiment Setup")

# Dataset Selector
dataset_name = st.sidebar.selectbox(
    "Choose a dataset:",
    ('CIFAR-10', 'CIFAR-100', 'STL10', 'FashionMNIST')
)

# K-shot Selector
initial_k_shot = st.sidebar.radio(
    "Initial Memory Size (k-shot):",
    (1, 5),
    captions=["Fastest, but less accurate", "Slower start, but more robust"]
)

# Load models and data based on selection
model, processor, device = load_clip_model()
train_dataset, CLASS_NAMES = load_dataset(dataset_name)

# --- State Management: Reset memory if dataset or settings change ---
force_reset = False
if 'dataset_name' not in st.session_state or st.session_state.dataset_name != dataset_name:
    force_reset = True
    st.session_state.dataset_name = dataset_name

if st.sidebar.button("Reset Memory", use_container_width=True):
    force_reset = True

if 'memory' not in st.session_state or force_reset:
    with st.spinner(f'Building new {initial_k_shot}-shot memory for {dataset_name}...'):
        st.session_state.memory = initialize_memory(initial_k_shot, train_dataset, model, processor, device)
    st.success(f"Memory initialized for {dataset_name} with {initial_k_shot} examples per class.")
    st.rerun()

# --- Main Page Layout ---
col_main, col_memory = st.columns([2, 1])

with col_main:
    st.header("Classifier")
    uploaded_file = st.file_uploader("Upload an image to classify", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        predicted_name, _, confidence = classify_image(image, model, processor, st.session_state.memory, CLASS_NAMES, device)
        st.metric("Model Prediction", f"{predicted_name}", f"Confidence: {confidence:.2%}")

        st.write("---")
        st.subheader("Is this prediction correct?")
        
        fb_col1, fb_col2 = st.columns(2)
        if fb_col1.button("✔️ Correct", use_container_width=True):
            st.success("Great! The model's knowledge is reinforced.")
        if fb_col2.button("❌ Incorrect", use_container_width=True):
            st.session_state.correction_mode = True

        if st.session_state.get('correction_mode'):
            st.warning("Please provide the correct label to update the model's memory.")
            correct_label_name = st.selectbox("Select the correct class:", CLASS_NAMES)
            
            if st.button("Submit Correction", use_container_width=True):
                correct_label_id = CLASS_NAMES.index(correct_label_name)
                with st.spinner("Updating memory..."):
                    with torch.no_grad():
                        image_input = processor(images=image, return_tensors="pt").to(device)
                        embedding = model.get_image_features(**image_input).squeeze(0)
                    st.session_state.memory.add_example(embedding, correct_label_id)
                
                st.success(f"Memory updated for class '{correct_label_name}'. Upload the image again to see the new prediction.")
                del st.session_state.correction_mode
                st.rerun()

with col_memory:
    st.header("Episodic Memory Status")
    memory_instance = st.session_state.get('memory')
    if memory_instance:
        all_labels = [label for label, embeddings in memory_instance.storage.items() for _ in embeddings]
        label_counts = Counter(all_labels)

        st.info(f"Total examples in memory: **{len(all_labels)}**")

        data = [{"Class": class_name, "Examples": label_counts.get(i, 0)} for i, class_name in enumerate(CLASS_NAMES)]
        st.dataframe(data, use_container_width=True, height=35 * len(CLASS_NAMES))
    else:
        st.write("Memory is not initialized yet.")