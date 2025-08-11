# File: clip_eval_utils.py
# (Only the load_dataset function needs to be changed)

import torch
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, STL10, ImageFolder
from torchvision import transforms

# (The Config class at the top of the file remains the same)
class Config:
    # --- Dataset Configuration ---
    # Set DATASET_NAME to one of: "CIFAR10", "CIFAR100", "SVHN", "FashionMNIST", "STL10", "ImageFolder"
    DATASET_NAME = "SVHN" # This is the setting that caused the error
    
    # If using "ImageFolder", set this to the root directory.
    # The structure must be: DATA_ROOT/train/class_name/images... and DATA_ROOT/test/class_name/images...
    # For TinyImageNet, you would download and unzip it, then point DATA_ROOT to the 'tiny-imagenet-200' folder.
    DATA_ROOT = "./data"

    # --- Model and Device Configuration ---
    MODEL_NAME = "openai/clip-vit-base-patch32"
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    BATCH_SIZE = 64


# --- 2. Flexible Data Loader (CORRECTED VERSION) ---

def load_dataset(name, root):
    """
    Loads the specified dataset by name.
    Handles various torchvision datasets and a generic ImageFolder structure.
    Returns: train_dataset, test_dataset, class_names
    """
    print(f"\nLoading dataset: {name} from {root}")
    transform = transforms.ToTensor()
    class_names = None  # Initialize class_names to None

    # --- torchvision datasets ---
    if name == "CIFAR10":
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=root, train=False, download=True, transform=transform)
        class_names = train_dataset.classes
    elif name == "CIFAR100":
        train_dataset = CIFAR100(root=root, train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root=root, train=False, download=True, transform=transform)
        class_names = train_dataset.classes
    elif name == "SVHN":
        train_dataset = SVHN(root=root, split='train', download=True, transform=transform)
        test_dataset = SVHN(root=root, split='test', download=True, transform=transform)
        train_dataset.targets = train_dataset.labels
        # --- FIX IS HERE ---
        # SVHN does not have a .classes attribute, so we define it manually.
        # The labels are integers 0-9. We convert them to strings for the text prompts.
        class_names = [str(i) for i in range(10)]
    elif name == "FashionMNIST":
        train_dataset = FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root=root, train=False, download=True, transform=transform)
        class_names = train_dataset.classes
    elif name == "STL10":
        train_dataset = STL10(root=root, split='train', download=True, transform=transform)
        test_dataset = STL10(root=root, split='test', download=True, transform=transform)
        train_dataset.targets = train_dataset.labels
        class_names = train_dataset.classes
    # --- Generic ImageFolder ---
    elif name == "ImageFolder":
        train_path = f"{root}/train"
        test_path = f"{root}/test"
        print(f"Loading from ImageFolder structure: {train_path} and {test_path}")
        train_dataset = ImageFolder(root=train_path, transform=transform)
        test_dataset = ImageFolder(root=test_path, transform=transform)
        train_dataset.targets = [sample[1] for sample in train_dataset.imgs]
        class_names = train_dataset.classes
    else:
        raise ValueError(f"Unknown dataset: '{name}'. Supported datasets are: "
                         "'CIFAR10', 'CIFAR100', 'SVHN', 'FashionMNIST', 'STL10', 'ImageFolder'.")
    
    if class_names is None:
        raise RuntimeError(f"Class names could not be determined for dataset {name}.")

    print(f"Dataset '{name}' loaded successfully.")
    print(f"Found {len(class_names)} classes: {class_names[:5]}...")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset, class_names