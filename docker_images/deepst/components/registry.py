from typing import Dict, Type
import torch.nn as nn

_ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {}
_GRAPH_ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {}
_DECODER_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_encoder(name: str):
    def decorator(cls):
        _ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def register_graph_encoder(name: str):
    def decorator(cls):
        _GRAPH_ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def register_decoder(name: str):
    def decorator(cls):
        _DECODER_REGISTRY[name] = cls
        return cls
    return decorator


def get_encoder(name: str) -> Type[nn.Module]:
    if name not in _ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(_ENCODER_REGISTRY.keys())}")
    return _ENCODER_REGISTRY[name]


def get_graph_encoder(name: str) -> Type[nn.Module]:
    if name not in _GRAPH_ENCODER_REGISTRY:
        raise ValueError(f"Unknown graph encoder: {name}. Available: {list(_GRAPH_ENCODER_REGISTRY.keys())}")
    return _GRAPH_ENCODER_REGISTRY[name]


def get_decoder(name: str) -> Type[nn.Module]:
    if name not in _DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder: {name}. Available: {list(_DECODER_REGISTRY.keys())}")
    return _DECODER_REGISTRY[name]
