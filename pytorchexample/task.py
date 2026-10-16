"""pytorchexample.task: Tabular Training and Evaluation Engines for FedSynAI.

Handles:
    E2  Standard FL   - use_dp=False, fedprox_mu=0.0
    E3  Non-IID FL    - fedprox_mu > 0.0  (FedProx proximal regularization)
    E4  DP-FL         - use_dp=True       (Opacus Differential Privacy)
    E5  FedSynAI      - use_dp=True, fedprox_mu > 0.0 (+ Synthetic Data from plugin)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from pytorchexample.privacy_engine import DPWrapper


def train_tabular(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    use_dp: bool = False,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    fedprox_mu: float = 0.0,
) -> tuple[float, float]:
    """Train a tabular model supporting all FedSynAI experiment modes.

    Args:
        net:              Model already loaded with global weights.
        trainloader:      Client's local DataLoader (features, labels) tuples.
        epochs:           Number of local training epochs.
        lr:               Learning rate.
        device:           torch.device for computation.
        use_dp:           Wrap with Opacus PrivacyEngine (E4/E5).
        noise_multiplier: Gaussian noise added to gradients (Opacus).
        max_grad_norm:    Per-sample gradient clip norm (Opacus).
        fedprox_mu:       FedProx proximal coefficient mu; 0.0 = standard FedAvg.

    Returns:
        (avg_train_loss: float, epsilon: float)
        epsilon is float('inf') when DP is disabled.
    """
    net.to(device)
    # Freeze a copy of the global weights for FedProx regularisation (E3/E5)
    global_params = [p.detach().clone() for p in net.parameters()]

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # DPWrapper is a no-op when enabled=False - same code path for all experiments
    dp = DPWrapper(
        enabled=use_dp,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
    )
    model, optimizer, trainloader = dp.prepare(net, optimizer, trainloader)
    model.train()

    running_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for features, labels in trainloader:
            if len(features) == 0:
                continue
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            # FedProx proximal term: pulls local weights toward global weights
            if fedprox_mu > 0.0:
                proximal_term = sum(
                    (w - w_t).norm(2)
                    for w, w_t in zip(model.parameters(), global_params)
                )
                loss = loss + (fedprox_mu / 2.0) * proximal_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1

    # If Opacus wrapped the model, copy weights back into the original net object
    if use_dp and hasattr(model, "_module"):
        net.load_state_dict(model._module.state_dict())

    avg_loss = running_loss / max(total_batches, 1)
    epsilon = dp.get_epsilon()
    return avg_loss, epsilon


def test_tabular(net: nn.Module, testloader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate a tabular model on a TensorDataset DataLoader.

    Works with any plugin that returns (X: FloatTensor, y: LongTensor) batches.
    Returns a dict with keys 'loss', 'accuracy', and 'f1' for the evaluation table.

    Args:
        net:        Trained nn.Module (e.g. TabularMLP from credit_risk plugin).
        testloader: DataLoader whose batches are (features, labels) tuples.
        device:     torch.device for inference.

    Returns:
        dict with float values for loss, accuracy, and f1.
    """
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            total_loss += criterion(outputs, labels).item()
            preds = torch.max(outputs, 1)[1]
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n_batches = max(len(testloader), 1)
    avg_loss = total_loss / n_batches
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}
