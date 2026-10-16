"""pytorchexample.plugins: Enterprise dataset plugin registry.

Importing this package is enough to register all built-in plugins.
Use load_plugin(name) to get an initialised FedSynAIPlugin instance.
"""

# ---- Import every plugin module so their @register_plugin decorators fire ----
from pytorchexample.plugins import credit_risk as _credit_risk  # noqa: F401

# Re-export the registry loader as the public API of this package
from pytorchexample.plugins.interfaces import FedSynAIPlugin, load_plugin

__all__ = ["FedSynAIPlugin", "load_plugin"]
