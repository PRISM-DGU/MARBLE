from typing import Dict, Type, Callable

ENCODER_IMPORTS: Dict[str, Callable] = {
    "encoder_stagate_gat": lambda: __import__("components.encoder_stagate_gat", fromlist=["Encoder"]).Encoder,
    "encoder_other": lambda: __import__("components.encoder_other", fromlist=["Encoder"]).Encoder,
}

DECODER_IMPORTS: Dict[str, Callable] = {
    "decoder_stagate_gat": lambda: __import__("components.decoder_stagate_gat", fromlist=["Decoder"]).Decoder,
    "decoder_other": lambda: __import__("components.decoder_other", fromlist=["Decoder"]).Decoder,
}

_ENCODER_CACHE: Dict[str, Type] = {}
_DECODER_CACHE: Dict[str, Type] = {}


def get_encoder(name: str) -> Type:
    if name in _ENCODER_CACHE:
        return _ENCODER_CACHE[name]

    if name not in ENCODER_IMPORTS:
        raise ValueError(f"Unknown encoder: '{name}'. Available: {list(ENCODER_IMPORTS.keys())}")

    cls = ENCODER_IMPORTS[name]()
    _ENCODER_CACHE[name] = cls
    return cls


def get_decoder(name: str) -> Type:
    if name in _DECODER_CACHE:
        return _DECODER_CACHE[name]

    if name not in DECODER_IMPORTS:
        raise ValueError(f"Unknown decoder: '{name}'. Available: {list(DECODER_IMPORTS.keys())}")

    cls = DECODER_IMPORTS[name]()
    _DECODER_CACHE[name] = cls
    return cls


def list_available_components() -> Dict[str, list]:
    return {
        "encoders": list(ENCODER_IMPORTS.keys()),
        "decoders": list(DECODER_IMPORTS.keys()),
    }
