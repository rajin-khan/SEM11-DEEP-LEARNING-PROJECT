# File: calibration.py
# Contains the logic for Temperature Scaling to calibrate model confidence.

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms  # <--- THIS LINE IS THE FIX

class CalibratedModel(nn.Module):
    """
    A wrapper for our prototypical network that adds a learnable temperature parameter.
    """
    def __init__(self, model, processor, memory, device):
        super(CalibratedModel, self).__init__()
        self.model = model
        self.processor = processor
        self.memory = memory
        self.device = device
        # Initialize temperature to 1 (no change)
        self.temperature = nn.Parameter(torch.ones(1, device=device))

    def forward(self, images):
        """Calculates logits for a batch of images using the memory prototypes."""
        with torch.no_grad():
            image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            query_embeddings = self.model.get_image_features(**image_inputs)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        prototypes, class_ids = self.memory.get_prototypes()
        prototypes = prototypes.to(self.device)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Create a mapping from original class_id to its index in the prototype tensor
        self.class_id_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        
        # Calculate logits (cosine similarity)
        logits = query_embeddings @ prototypes.T
        return logits

    def temperature_scale(self, logits):
        """Applies temperature scaling to logits."""
        return logits / self.temperature

    def set_temperature(self, validation_loader):
        """
        Tunes the temperature parameter on a validation set.
        """
        self.eval()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)

        all_logits = []
        all_labels = []

        # 1. Collect all logits and labels from the validation set
        print("Collecting logits from validation set for calibration...")
        with torch.no_grad():
            for images, labels in validation_loader:
                pil_images = [transforms.ToPILImage()(img) for img in images]
                logits = self.forward(pil_images)
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits).to(self.device)
        all_labels = torch.cat(all_labels).to(self.device)
        
        # Map global labels to the local indices of the prototypes
        mapped_labels = torch.tensor([self.class_id_to_idx[lbl.item()] for lbl in all_labels], device=self.device)

        # 2. Optimize the temperature parameter
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(all_logits), mapped_labels)
            loss.backward()
            return loss
        
        print("Optimizing temperature...")
        optimizer.step(eval)
        
        print(f"Optimal temperature found: {self.temperature.item():.4f}")
        return self