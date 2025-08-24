# File: agent.py
# Defines the ReadinessAgent and its evaluation metrics.

import numpy as np
from sklearn.metrics import confusion_matrix

class ReadinessAgent:
    def __init__(self, threshold=0.9):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")
        self.threshold = threshold
        print(f"ReadinessAgent initialized with confidence threshold: {self.threshold:.2%}")

    def decide(self, confidence):
        """Makes a decision based on the confidence score."""
        if confidence >= self.threshold:
            return "Deploy"
        else:
            return "Request Feedback"

    @staticmethod
    def get_metrics(model_predictions, true_labels, agent_decisions):
        """Calculates key performance metrics for the agent."""
        model_correct = (model_predictions == true_labels)
        
        # --- Deployment Decision Accuracy ---
        # A decision is "correct" if:
        # 1. The agent DEPLOYED and the model was RIGHT.
        # 2. The agent REQUESTED FEEDBACK and the model was WRONG.
        correct_deploys = np.sum((agent_decisions == "Deploy") & model_correct)
        correct_requests = np.sum((agent_decisions == "Request Feedback") & ~model_correct)
        decision_accuracy = (correct_deploys + correct_requests) / len(true_labels)

        # --- False Positive Rate (FPR) ---
        # Of all the times the model was actually WRONG, what fraction did the agent confidently DEPLOY?
        total_model_errors = np.sum(~model_correct)
        incorrect_deploys = np.sum((agent_decisions == "Deploy") & ~model_correct)
        fpr = incorrect_deploys / total_model_errors if total_model_errors > 0 else 0.0

        return {
            "Deployment Decision Accuracy": decision_accuracy,
            "False Positive Rate (FPR)": fpr,
            "Total Model Accuracy": np.mean(model_correct),
            "Total Deployed": np.mean(agent_decisions == "Deploy"),
        }