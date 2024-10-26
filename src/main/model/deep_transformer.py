"""
The actual decoder implementation is specified here.
"""

import virtual_layers
import registry

# Registry imports

# Setup a few additional registries for the builder

feedforward_registry = registry.TorchLayerRegistry[virtual_layers.VirtualFeedforward]("Feedforward",
                                                                        virtual_layers.VirtualFeedforward)
feedforward_registry.register_class("Default", virtual_layers.VirtualFeedforward)

