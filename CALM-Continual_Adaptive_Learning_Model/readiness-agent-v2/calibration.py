# File: calibration.py (IMPROVED VERSION)
# Enhanced calibration with debugging and multiple calibration methods

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class CalibratedModel(nn.Module):
    """
    Enhanced wrapper with multiple calibration methods and debugging capabilities.
    """
    def __init__(self, model, processor, memory, device):
        super(CalibratedModel, self).__init__()
        self.model = model
        self.processor = processor
        self.memory = memory
        self.device = device
        # Initialize temperature to 1 (no change)
        self.temperature = nn.Parameter(torch.ones(1, device=device))
        
        # Alternative calibration methods
        self.isotonic_calibrator = None
        self.platt_calibrator = None
        self.calibration_method = "temperature"  # Default method

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

    def get_calibrated_confidences(self, logits):
        """Get calibrated confidence scores based on the selected method."""
        if self.calibration_method == "temperature":
            scaled_logits = logits / self.temperature
            probabilities = torch.softmax(scaled_logits, dim=1)
            confidences, _ = torch.max(probabilities, dim=1)
            return confidences
        
        elif self.calibration_method == "isotonic":
            probabilities = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            max_probs_np = max_probs.detach().cpu().numpy()
            if self.isotonic_calibrator is not None:
                calibrated_confidences = self.isotonic_calibrator.predict(max_probs_np)
                return torch.tensor(calibrated_confidences, device=self.device)
            return max_probs
        
        elif self.calibration_method == "platt":
            probabilities = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            max_probs_np = max_probs.detach().cpu().numpy().reshape(-1, 1)
            if self.platt_calibrator is not None:
                calibrated_probs = self.platt_calibrator.predict_proba(max_probs_np)[:, 1]
                return torch.tensor(calibrated_probs, device=self.device)
            return max_probs
        
        else:  # No calibration
            probabilities = torch.softmax(logits, dim=1)
            confidences, _ = torch.max(probabilities, dim=1)
            return confidences

    def temperature_scale(self, logits):
        """Applies temperature scaling to logits."""
        return logits / self.temperature

    def analyze_confidence_distribution(self, validation_loader, title="Confidence Distribution"):
        """Analyze and plot confidence distributions before and after calibration."""
        self.eval()
        raw_confidences = []
        calibrated_confidences = []
        accuracies = []
        
        print(f"\n=== {title} ===")
        
        with torch.no_grad():
            for images, labels in validation_loader:
                pil_images = [transforms.ToPILImage()(img) for img in images]
                logits = self.forward(pil_images)
                
                # Raw confidences
                raw_probs = torch.softmax(logits, dim=1)
                raw_conf, raw_preds = torch.max(raw_probs, dim=1)
                
                # Calibrated confidences
                cal_conf = self.get_calibrated_confidences(logits)
                
                # Map predictions back to original labels
                class_ids = sorted(self.memory.storage.keys())
                mapped_preds = torch.tensor([class_ids[p] for p in raw_preds], device=self.device)
                
                # Calculate accuracy
                correct = (mapped_preds.cpu() == labels).float()
                
                raw_confidences.extend(raw_conf.cpu().numpy())
                calibrated_confidences.extend(cal_conf.cpu().numpy())
                accuracies.extend(correct.numpy())
        
        raw_confidences = np.array(raw_confidences)
        calibrated_confidences = np.array(calibrated_confidences)
        accuracies = np.array(accuracies)
        
        print(f"Raw confidence stats:")
        print(f"  Mean: {np.mean(raw_confidences):.4f}")
        print(f"  Std: {np.std(raw_confidences):.4f}")
        print(f"  Min: {np.min(raw_confidences):.4f}")
        print(f"  Max: {np.max(raw_confidences):.4f}")
        
        print(f"Calibrated confidence stats:")
        print(f"  Mean: {np.mean(calibrated_confidences):.4f}")
        print(f"  Std: {np.std(calibrated_confidences):.4f}")
        print(f"  Min: {np.min(calibrated_confidences):.4f}")
        print(f"  Max: {np.max(calibrated_confidences):.4f}")
        
        # Calculate Expected Calibration Error (ECE)
        ece = self.calculate_ece(calibrated_confidences, accuracies)
        print(f"Expected Calibration Error: {ece:.4f}")
        
        return raw_confidences, calibrated_confidences, accuracies

    def calculate_ece(self, confidences, accuracies, n_bins=10):
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

    def set_temperature_robust(self, validation_loader, method="temperature"):
        """
        Enhanced temperature setting with multiple methods and extensive debugging.
        """
        self.calibration_method = method
        self.eval()
        
        all_logits = []
        all_labels = []
        all_correct = []

        # 1. Collect all logits and labels from the validation set
        print(f"Collecting logits from validation set for {method} calibration...")
        with torch.no_grad():
            for images, labels in validation_loader:
                pil_images = [transforms.ToPILImage()(img) for img in images]
                logits = self.forward(pil_images)
                
                # Get predictions for accuracy calculation
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(probs, dim=1)
                class_ids = sorted(self.memory.storage.keys())
                mapped_preds = torch.tensor([class_ids[p] for p in preds], device=self.device)
                correct = (mapped_preds.cpu() == labels).float()
                
                all_logits.append(logits)
                all_labels.append(labels)
                all_correct.append(correct)
        
        all_logits = torch.cat(all_logits).to(self.device)
        all_labels = torch.cat(all_labels).to(self.device)
        all_correct = torch.cat(all_correct).to(self.device)
        
        # Map global labels to the local indices of the prototypes
        mapped_labels = torch.tensor([self.class_id_to_idx[lbl.item()] for lbl in all_labels], device=self.device)

        if method == "temperature":
            self._calibrate_temperature(all_logits, mapped_labels)
        elif method == "isotonic":
            self._calibrate_isotonic(all_logits, all_correct)
        elif method == "platt":
            self._calibrate_platt(all_logits, all_correct)
        
        return self

    def _calibrate_temperature(self, all_logits, mapped_labels):
        """Calibrate using temperature scaling with enhanced diagnostics."""
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Check initial loss
        initial_loss = nll_criterion(all_logits, mapped_labels)
        print(f"Initial NLL loss: {initial_loss.item():.4f}")
        
        # Try multiple optimization strategies
        best_temperature = 1.0
        best_loss = float('inf')
        
        for lr in [0.01, 0.001, 0.1]:
            for max_iter in [50, 100, 200]:
                # Reset temperature
                self.temperature.data.fill_(1.0)
                optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

                def eval_fn():
                    optimizer.zero_grad()
                    loss = nll_criterion(self.temperature_scale(all_logits), mapped_labels)
                    loss.backward()
                    return loss

                try:
                    optimizer.step(eval_fn)
                    final_loss = nll_criterion(self.temperature_scale(all_logits), mapped_labels)
                    
                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_temperature = self.temperature.item()
                        
                except:
                    continue
        
        # Set the best temperature found
        self.temperature.data.fill_(best_temperature)
        print(f"Best temperature found: {best_temperature:.4f}")
        print(f"Final NLL loss: {best_loss:.4f}")
        
        # Warn if temperature is extreme
        if best_temperature > 10:
            print(f"WARNING: Very high temperature ({best_temperature:.2f}) detected!")
            print("This may indicate overfitting or poor model calibration.")
        elif best_temperature < 0.1:
            print(f"WARNING: Very low temperature ({best_temperature:.2f}) detected!")
            print("This may lead to overconfident predictions.")

    def _calibrate_isotonic(self, all_logits, all_correct):
        """Calibrate using Isotonic Regression."""
        probs = torch.softmax(all_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        
        max_probs_np = max_probs.detach().cpu().numpy()
        correct_np = all_correct.cpu().numpy()
        
        self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_calibrator.fit(max_probs_np, correct_np)
        
        print("Isotonic calibration completed.")

    def _calibrate_platt(self, all_logits, all_correct):
        """Calibrate using Platt Scaling (Logistic Regression)."""
        probs = torch.softmax(all_logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        
        max_probs_np = max_probs.detach().cpu().numpy().reshape(-1, 1)
        correct_np = all_correct.cpu().numpy()
        
        self.platt_calibrator = LogisticRegression()
        self.platt_calibrator.fit(max_probs_np, correct_np)
        
        print("Platt scaling calibration completed.")

    # Keep backward compatibility
    def set_temperature(self, validation_loader):
        """Backward compatibility wrapper."""
        return self.set_temperature_robust(validation_loader, method="temperature")