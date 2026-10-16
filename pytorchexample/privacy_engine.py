"""pytorchexample.privacy_engine: Differential Privacy via Opacus.

Experiment 4 (DP-FL) and Experiment 5 (FedSynAI) both use this module.
Toggle via pyproject.toml:
    use-dp = true
    noise-multiplier = 1.1
    max-grad-norm = 1.0
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


class DPWrapper:
    """Stateful Differential Privacy context used by train_tabular.

    When ``enabled=False`` (E2/E3), every method is a no-op so train_tabular
    needs zero conditional branches — the same code path runs for all experiments.

    Usage::
        dp = DPWrapper(enabled=use_dp, max_grad_norm=1.0, noise_multiplier=1.1)
        model, optimizer, loader = dp.prepare(model, optimizer, loader)
        # ... training loop ...
        epsilon = dp.get_epsilon()   # float("inf") when DP is off
    """

    def __init__(
        self,
        enabled: bool = False,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        target_delta: float = 1e-5,
    ) -> None:
        self.enabled = enabled
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self._privacy_engine = None

    def prepare(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """Attach Opacus PrivacyEngine; returns unchanged objects if DP is off."""
        if not self.enabled:
            return model, optimizer, data_loader

        try:
            from opacus import PrivacyEngine
        except ImportError as exc:
            raise ImportError(
                "opacus is required for DP training. "
                "Install it with: pip install opacus>=1.5.0"
            ) from exc

        self._privacy_engine = PrivacyEngine()
        dp_model, dp_optimizer, dp_loader, *_ = self._privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        print(
            f"[DP] PrivacyEngine attached | "
            f"noise={self.noise_multiplier} | "
            f"clip={self.max_grad_norm} | "
            f"delta={self.target_delta}"
        )
        return dp_model, dp_optimizer, dp_loader

    def get_epsilon(self) -> float:
        """Return epsilon spent so far; ``float('inf')`` when DP is disabled."""
        if not self.enabled or self._privacy_engine is None:
            return float("inf")
        epsilon = self._privacy_engine.get_epsilon(delta=self.target_delta)
        print(f"[DP] eps = {epsilon:.4f}  (delta={self.target_delta})")
        return float(epsilon)


def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    target_delta: float = 1e-5,
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
    """Wrap model, optimizer, and dataloader with Opacus PrivacyEngine.

    Args:
        model:            Any nn.Module (e.g., the tabular MLP from task.py).
        optimizer:        A standard PyTorch optimizer.
        data_loader:      The client's training DataLoader.
        noise_multiplier: Controls the amount of Gaussian noise added (sigma).
                          Higher -> more privacy, less accuracy.
        max_grad_norm:    Per-sample gradient clipping norm (C).
        target_delta:     Privacy budget delta (should be << 1/dataset_size).

    Returns:
        Tuple of (dp_model, dp_optimizer, dp_data_loader) - drop-in replacements.

    Raises:
        ImportError: If opacus is not installed.
    """
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise ImportError(
            "opacus is required for DP training. "
            "Install it with: pip install opacus>=1.5.0"
        ) from exc

    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader, *_ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    print(
        f"[DP] PrivacyEngine attached | "
        f"noise_multiplier={noise_multiplier} | "
        f"max_grad_norm={max_grad_norm} | "
        f"delta={target_delta}"
    )
    return dp_model, dp_optimizer, dp_loader


def get_epsilon(
    privacy_engine,
    target_delta: float = 1e-5,
) -> float:
    """Compute the current (eps, delta)-DP budget spent so far.

    Call this after training to log privacy cost per round.
    """
    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    print(f"[DP] Privacy spent: eps={epsilon:.4f} at delta={target_delta}")
    return epsilon

