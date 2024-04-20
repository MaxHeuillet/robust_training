from lora import LoRAParametrization
from torch import nn


def apply_to_lora(fn):
    """apply a function to LoRAParametrization layers, designed to be used with model.apply"""

    def apply_fn(layer):
        if isinstance(layer, LoRAParametrization):
            fn(layer)

    return apply_fn

# ------------------- helper function for collecting parameters for training/saving -------------------



