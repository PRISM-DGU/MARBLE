"""DeepDR Component Registry - Dynamic component loading"""

from typing import Dict, Type, Callable

# Lazy imports for drug encoders
DRUG_ENCODER_IMPORTS: Dict[str, Callable] = {
    "drug_encoder_deepdr": lambda: __import__("components.drug_encoder_deepdr", fromlist=["DrugEncoder"]).DrugEncoder,
    "drug_encoder_other": lambda: __import__("components.drug_encoder_other", fromlist=["DrugEncoder"]).DrugEncoder,
}

# Lazy imports for cell encoders
CELL_ENCODER_IMPORTS: Dict[str, Callable] = {
    "cell_encoder_deepdr": lambda: __import__("components.cell_encoder_deepdr", fromlist=["CellEncoder"]).CellEncoder,
    "cell_encoder_other": lambda: __import__("components.cell_encoder_other", fromlist=["CellEncoder"]).CellEncoder,
}

# Lazy imports for decoders
DECODER_IMPORTS: Dict[str, Callable] = {
    "decoder_deepdr": lambda: __import__("components.decoder_deepdr", fromlist=["Decoder"]).Decoder,
    "decoder_other": lambda: __import__("components.decoder_other", fromlist=["Decoder"]).Decoder,
}

# Lazy imports for data loaders
DRUG_DATA_LOADER_IMPORTS: Dict[str, Callable] = {
    "drug_encoder_deepdr": lambda: __import__("components.drug_encoder_deepdr", fromlist=["DrugDataLoader"]).DrugDataLoader,
    "drug_encoder_other": lambda: __import__("components.drug_encoder_other", fromlist=["DrugDataLoader"]).DrugDataLoader,
}

CELL_DATA_LOADER_IMPORTS: Dict[str, Callable] = {
    "cell_encoder_deepdr": lambda: __import__("components.cell_encoder_deepdr", fromlist=["CellDataLoader"]).CellDataLoader,
    "cell_encoder_other": lambda: __import__("components.cell_encoder_other", fromlist=["CellDataLoader"]).CellDataLoader,
}

# Caches
_DRUG_ENCODER_CACHE: Dict[str, Type] = {}
_CELL_ENCODER_CACHE: Dict[str, Type] = {}
_DECODER_CACHE: Dict[str, Type] = {}


def get_drug_encoder(name: str) -> Type:
    """Get drug encoder class by name"""
    if name in _DRUG_ENCODER_CACHE:
        return _DRUG_ENCODER_CACHE[name]

    if name not in DRUG_ENCODER_IMPORTS:
        raise ValueError(f"Unknown drug encoder: '{name}'. Available: {list(DRUG_ENCODER_IMPORTS.keys())}")

    cls = DRUG_ENCODER_IMPORTS[name]()
    _DRUG_ENCODER_CACHE[name] = cls
    return cls


def get_cell_encoder(name: str) -> Type:
    """Get cell encoder class by name"""
    if name in _CELL_ENCODER_CACHE:
        return _CELL_ENCODER_CACHE[name]

    if name not in CELL_ENCODER_IMPORTS:
        raise ValueError(f"Unknown cell encoder: '{name}'. Available: {list(CELL_ENCODER_IMPORTS.keys())}")

    cls = CELL_ENCODER_IMPORTS[name]()
    _CELL_ENCODER_CACHE[name] = cls
    return cls


def get_decoder(name: str) -> Type:
    """Get decoder class by name"""
    if name in _DECODER_CACHE:
        return _DECODER_CACHE[name]

    if name not in DECODER_IMPORTS:
        raise ValueError(f"Unknown decoder: '{name}'. Available: {list(DECODER_IMPORTS.keys())}")

    cls = DECODER_IMPORTS[name]()
    _DECODER_CACHE[name] = cls
    return cls


def get_drug_data_loader(encoder_type: str) -> Type:
    """Get drug data loader class by encoder type"""
    if encoder_type not in DRUG_DATA_LOADER_IMPORTS:
        raise ValueError(f"Unknown drug encoder type for data loader: '{encoder_type}'. Available: {list(DRUG_DATA_LOADER_IMPORTS.keys())}")

    return DRUG_DATA_LOADER_IMPORTS[encoder_type]()


def get_cell_data_loader(encoder_type: str) -> Type:
    """Get cell data loader class by encoder type"""
    if encoder_type not in CELL_DATA_LOADER_IMPORTS:
        raise ValueError(f"Unknown cell encoder type for data loader: '{encoder_type}'. Available: {list(CELL_DATA_LOADER_IMPORTS.keys())}")

    return CELL_DATA_LOADER_IMPORTS[encoder_type]()


def list_available_components() -> Dict[str, list]:
    """List all available components"""
    return {
        "drug_encoders": list(DRUG_ENCODER_IMPORTS.keys()),
        "cell_encoders": list(CELL_ENCODER_IMPORTS.keys()),
        "decoders": list(DECODER_IMPORTS.keys()),
    }
