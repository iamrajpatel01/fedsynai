"""pytorchexample.security_audit: Privacy Auditing via Membership Inference Attack.

Uses a confidence-score-based shadow-model MIA to estimate how much a trained
model reveals about its training data. Run this after each experiment to
populate the Audit column in your results table.

Typical usage (post-training in server_app.py or a standalone script):
    from pytorchexample.security_audit import MembershipInferenceAuditor
    auditor = MembershipInferenceAuditor(model, device)
    auc = auditor.run(train_loader, test_loader)
    print(f"MIA AUC: {auc:.4f}")   # 0.5 = perfect privacy, 1.0 = full leakage
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class MembershipInferenceAuditor:
    """Lightweight confidence-score Membership Inference Attack (MIA).

    Strategy
    --------
    Members (training samples) tend to have higher model confidence than
    non-members (held-out samples).  We threshold on the maximum softmax
    probability and compute the AUC of this binary classifier.

    This is the metric used in the FedSynAI paper experiments to compare
    privacy leakage across E2 (no DP), E4 (DP-FL), and E5 (FedSynAI).
    """

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _get_confidences(self, loader: DataLoader) -> np.ndarray:
        """Return max-softmax confidence for every sample in loader."""
        confidences = []
        for batch in loader:
            # Support both dict-style (flwr-datasets) and tuple-style loaders
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(self.device)
            elif isinstance(batch, dict):
                # Tabular: features stored under key "features" or first non-label key
                feature_key = next(k for k in batch if k != "label")
                inputs = batch[feature_key].to(self.device)
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            max_conf = probs.max(dim=1).values.cpu().numpy()
            confidences.extend(max_conf.tolist())
        return np.array(confidences)

    def run(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> float:
        """Compute the MIA AUC score.

        Args:
            member_loader:     DataLoader of *training* samples (label = 1 = member).
            non_member_loader: DataLoader of *held-out* samples (label = 0 = non-member).

        Returns:
            AUC score in [0, 1].  0.5 ≈ random chance (good privacy).
        """
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for MIA auditing. "
                "Install it with: pip install scikit-learn>=1.5.0"
            ) from exc

        member_conf = self._get_confidences(member_loader)
        non_member_conf = self._get_confidences(non_member_loader)

        scores = np.concatenate([member_conf, non_member_conf])
        labels = np.concatenate([
            np.ones(len(member_conf)),
            np.zeros(len(non_member_conf)),
        ])

        auc = roc_auc_score(labels, scores)
        print(
            f"[SecurityAudit] MIA AUC = {auc:.4f} "
            f"(0.50 = perfect privacy | 1.00 = full leakage)"
        )
        return auc
