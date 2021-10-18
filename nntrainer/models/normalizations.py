"""
Normalization functions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import paddle
from paddle import nn

import nntrainer.utils
from nntrainer.typext import ConfigClass, ConstantHolder


def make_normalization_module(normalized_shape: Union[int, List[int], any], name: str,
                              cfg: Optional[NormalizationConfig] = None) -> paddle.nn.Layer:
    """
    Get normalization module instance given by name and config object.

    Args:
        normalized_shape: Input shape from an expected input.
        name:
        cfg: Hyperparameter config

    Returns:
        Normalization module instance.
    """
    if cfg is None:
        # set all module hyperparameters to default values
        cfg = NormalizationConfig(name)

    # create the module instance
    if name == NormalizationConst.NONE:
        return nn.Layer()
    if name == NormalizationConst.LAYERNORM_PYTORCH:
        return nn.LayerNorm(normalized_shape, epsilon=cfg.eps, elementwise_affine=cfg.affine)
    if name == NormalizationConst.LAYERNORM_COOT:
        return LayerNormalization(normalized_shape, epsilon=cfg.eps)
    raise NotImplementedError(f"Normalization {name} not found.")


class NormalizationConst(ConstantHolder):
    """
    Define normalization module names.
    """
    NONE = nntrainer.utils.NONE
    LAYERNORM_PYTORCH = "layernorm_pytorch"
    LAYERNORM_COOT = "layernorm_coot"


class NormalizationConfig(ConfigClass):
    """
    Normalization config object. Stores hyperparameters.

    Examples:
        >>> NormalizationConfig("layernorm_coot")
        >>> NormalizationConfig({"name": "layernorm_coot", "affine": "false"})

    Args:
        name_or_config: Either provide string name of the activation function (e.g. "layernorm") or a dict with name and
            hyperparameters (e.g. {"name": "layernorm", "epsilon": 1e-6})
    """

    def __init__(self, name_or_config: Union[str, Dict[str, Any]]):
        # Determine if configuration is given by a string name or a config dict.
        if isinstance(name_or_config, str):
            config: Dict[str, Any] = {}
            self.name = name_or_config
        elif isinstance(name_or_config, dict):
            config = name_or_config
            self.name = config.pop("name")
        else:
            raise ValueError(f"Type {name_or_config} not understood.")
        # Set optional fields
        self.eps: float = config.pop("eps", 1e-6)
        self.affine: bool = config.pop("affine", True)
        # StochNorm
        self.momentum: float = config.pop("momentum", 0.1)
        self.track_running_stats = config.pop("track_running_stats", True)


# ---------- Module implementations. ----------

class LayerNormalization(paddle.nn.Layer):
    """
    Layer Normalization - Normalize across features instead of across the
    batch like in BatchNorm. Independent of batch size.

    Different results from the PyTorch implementation.
    """
    def __init__(self, normalized_shape: Union[int, List[int], any], epsilon: float = 1e-6):
        super().__init__()

        x = paddle.zeros([normalized_shape], dtype="float32")
        self.gain = paddle.create_parameter(shape=x.shape,
                                            dtype=str(x.numpy().dtype),
                                            default_initializer=paddle.nn.initializer.Assign(x))
        self.gain.stop_gradient = True
        x=paddle.zeros_like(x)
        self.gain = paddle.create_parameter(shape=x.shape,
                                            dtype=str(x.numpy().dtype),
                                            default_initializer=paddle.nn.initializer.Assign(x))
        self.gain.stop_gradient = True
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias
