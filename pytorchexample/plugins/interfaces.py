"""pytorchexample.plugins.interfaces: Abstract base for all FedSynAI dataset plugins.

Every plugin (credit_risk, hospital_readmission, fraud_detection, …) must
implement this interface so that task.py can load any plugin by name without
any conditional logic — just:

    from pytorchexample.plugins import load_plugin
    plugin = load_plugin(plugin_name)           # e.g. "credit_risk"
    model   = plugin.build_model()
    train_l, val_l = plugin.load_partition(pid, n_partitions, batch_size, non_iid)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


class FedSynAIPlugin(ABC):
    """Contract every enterprise dataset plugin must satisfy."""

    # ------------------------------------------------------------------ #
    # Model                                                                 #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Return an initialised (untrained) PyTorch model for this task.

        Examples:
            - Credit Risk -> MLP with dropout
            - Medical Imaging -> ResNet-18 (if torchvision is re-added)
        """
        ...

    # ------------------------------------------------------------------ #
    # Data                                                                  #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def load_partition(
        self,
        partition_id: int,
        num_partitions: int,
        batch_size: int,
        non_iid: bool = False,
        use_synthetic: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        """Return (train_loader, val_loader) for the given client partition.

        Args:
            partition_id:   Zero-indexed client ID.
            num_partitions: Total number of clients (e.g., 10).
            batch_size:     Mini-batch size passed from run_config.
            non_iid:        If True, apply Dirichlet label-skew partitioning (α=0.5).
            use_synthetic:  If True (E5), augment training split with CTGAN-generated
                            synthetic rows before returning the DataLoader.

        Returns:
            Tuple of (train DataLoader, validation DataLoader).
        """
        ...

    @abstractmethod
    def load_centralized_dataset(self, batch_size: int = 128) -> DataLoader:
        """Return a DataLoader over the full held-out test set.

        Used by server_app.py for global evaluation after each round.
        """
        ...

    # ------------------------------------------------------------------ #
    # Meta                                                                  #
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name, e.g. ``'credit_risk'``."""
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes the model must produce."""
        ...

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Number of input features after preprocessing."""
        ...


# ------------------------------------------------------------------ #
# Plugin registry                                                      #
# ------------------------------------------------------------------ #

_REGISTRY: dict[str, type[FedSynAIPlugin]] = {}


def register_plugin(plugin_cls: type[FedSynAIPlugin]) -> type[FedSynAIPlugin]:
    """Class decorator to register a plugin by its name property."""
    # Instantiate temporarily just to read the name; subclasses must have
    # a no-arg __init__ that does NOT download data yet.
    _REGISTRY[plugin_cls._name] = plugin_cls  # type: ignore[attr-defined]
    return plugin_cls


def load_plugin(plugin_name: str) -> FedSynAIPlugin:
    """Instantiate and return the plugin registered under plugin_name.

    Args:
        plugin_name: Value of ``plugin-name`` in pyproject.toml, e.g. ``"credit_risk"``.

    Raises:
        KeyError: If no plugin with that name has been registered.
    """
    if plugin_name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "<none registered>"
        raise KeyError(
            f"Unknown plugin '{plugin_name}'. "
            f"Available plugins: {available}"
        )
    return _REGISTRY[plugin_name]()
