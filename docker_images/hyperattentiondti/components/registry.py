"""Component Registry - Lazy Import for HyperAttentionDTI components

Usage:
    from components import get_drug_encoder, get_protein_encoder, get_decoder
    from components import get_drug_data_loader, get_protein_data_loader

    # Get class dynamically from config type
    DrugEncoderClass = get_drug_encoder("drug_encoder_hyperattentiondti")
    encoder = DrugEncoderClass(config)

    # Data Loader
    DrugDataLoaderClass = get_drug_data_loader("drug_encoder_hyperattentiondti")
    loader = DrugDataLoaderClass(config, base_path)
"""

from typing import Dict, Type, Callable

# ============================================================================
# Lazy Import Mappings
# ============================================================================

DRUG_ENCODER_IMPORTS: Dict[str, Callable] = {
    "drug_encoder_hyperattentiondti": lambda: __import__("components.drug_encoder_hyperattentiondti", fromlist=["DrugEncoder"]).DrugEncoder,
    "drug_encoder_other": lambda: __import__("components.drug_encoder_other", fromlist=["DrugEncoder"]).DrugEncoder,
}

PROTEIN_ENCODER_IMPORTS: Dict[str, Callable] = {
    "protein_encoder_hyperattentiondti": lambda: __import__("components.protein_encoder_hyperattentiondti", fromlist=["ProteinEncoder"]).ProteinEncoder,
    "protein_encoder_other": lambda: __import__("components.protein_encoder_other", fromlist=["ProteinEncoder"]).ProteinEncoder,
}

DECODER_IMPORTS: Dict[str, Callable] = {
    "decoder_hyperattentiondti": lambda: __import__("components.decoder_hyperattentiondti", fromlist=["Decoder"]).Decoder,
    "decoder_other": lambda: __import__("components.decoder_other", fromlist=["Decoder"]).Decoder,
}

DRUG_DATA_LOADER_IMPORTS: Dict[str, Callable] = {
    "drug_encoder_hyperattentiondti": lambda: __import__("components.drug_encoder_hyperattentiondti", fromlist=["DrugDataLoader"]).DrugDataLoader,
    "drug_encoder_other": lambda: __import__("components.drug_encoder_other", fromlist=["DrugDataLoader"]).DrugDataLoader,
}

PROTEIN_DATA_LOADER_IMPORTS: Dict[str, Callable] = {
    "protein_encoder_hyperattentiondti": lambda: __import__("components.protein_encoder_hyperattentiondti", fromlist=["ProteinDataLoader"]).ProteinDataLoader,
    "protein_encoder_other": lambda: __import__("components.protein_encoder_other", fromlist=["ProteinDataLoader"]).ProteinDataLoader,
}

# Cache for already-loaded classes
_DRUG_ENCODER_CACHE: Dict[str, Type] = {}
_PROTEIN_ENCODER_CACHE: Dict[str, Type] = {}
_DECODER_CACHE: Dict[str, Type] = {}
_DRUG_DATA_LOADER_CACHE: Dict[str, Type] = {}
_PROTEIN_DATA_LOADER_CACHE: Dict[str, Type] = {}


# ============================================================================
# Getter Functions (Lazy Import)
# ============================================================================

def get_drug_encoder(name: str) -> Type:
    """Get drug encoder class by name (lazy import)"""
    if name in _DRUG_ENCODER_CACHE:
        return _DRUG_ENCODER_CACHE[name]

    if name not in DRUG_ENCODER_IMPORTS:
        available = list(DRUG_ENCODER_IMPORTS.keys())
        raise ValueError(f"Unknown drug_encoder: '{name}'. Available: {available}")

    cls = DRUG_ENCODER_IMPORTS[name]()
    _DRUG_ENCODER_CACHE[name] = cls
    return cls


def get_protein_encoder(name: str) -> Type:
    """Get protein encoder class by name (lazy import)"""
    if name in _PROTEIN_ENCODER_CACHE:
        return _PROTEIN_ENCODER_CACHE[name]

    if name not in PROTEIN_ENCODER_IMPORTS:
        available = list(PROTEIN_ENCODER_IMPORTS.keys())
        raise ValueError(f"Unknown protein_encoder: '{name}'. Available: {available}")

    cls = PROTEIN_ENCODER_IMPORTS[name]()
    _PROTEIN_ENCODER_CACHE[name] = cls
    return cls


def get_decoder(name: str) -> Type:
    """Get decoder class by name (lazy import)"""
    if name in _DECODER_CACHE:
        return _DECODER_CACHE[name]

    if name not in DECODER_IMPORTS:
        available = list(DECODER_IMPORTS.keys())
        raise ValueError(f"Unknown decoder: '{name}'. Available: {available}")

    cls = DECODER_IMPORTS[name]()
    _DECODER_CACHE[name] = cls
    return cls


def get_drug_data_loader(name: str) -> Type:
    """Get drug data loader class by name (lazy import)"""
    if name in _DRUG_DATA_LOADER_CACHE:
        return _DRUG_DATA_LOADER_CACHE[name]

    if name not in DRUG_DATA_LOADER_IMPORTS:
        available = list(DRUG_DATA_LOADER_IMPORTS.keys())
        raise ValueError(f"Unknown drug_data_loader: '{name}'. Available: {available}")

    cls = DRUG_DATA_LOADER_IMPORTS[name]()
    _DRUG_DATA_LOADER_CACHE[name] = cls
    return cls


def get_protein_data_loader(name: str) -> Type:
    """Get protein data loader class by name (lazy import)"""
    if name in _PROTEIN_DATA_LOADER_CACHE:
        return _PROTEIN_DATA_LOADER_CACHE[name]

    if name not in PROTEIN_DATA_LOADER_IMPORTS:
        available = list(PROTEIN_DATA_LOADER_IMPORTS.keys())
        raise ValueError(f"Unknown protein_data_loader: '{name}'. Available: {available}")

    cls = PROTEIN_DATA_LOADER_IMPORTS[name]()
    _PROTEIN_DATA_LOADER_CACHE[name] = cls
    return cls


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_components() -> Dict[str, list]:
    """List all registered components (without importing)"""
    return {
        "drug_encoders": list(DRUG_ENCODER_IMPORTS.keys()),
        "protein_encoders": list(PROTEIN_ENCODER_IMPORTS.keys()),
        "decoders": list(DECODER_IMPORTS.keys()),
        "drug_data_loaders": list(DRUG_DATA_LOADER_IMPORTS.keys()),
        "protein_data_loaders": list(PROTEIN_DATA_LOADER_IMPORTS.keys()),
    }


def is_registered(component_type: str, name: str) -> bool:
    """Check if a component is registered"""
    registry_map = {
        "drug_encoder": DRUG_ENCODER_IMPORTS,
        "protein_encoder": PROTEIN_ENCODER_IMPORTS,
        "decoder": DECODER_IMPORTS,
        "drug_data_loader": DRUG_DATA_LOADER_IMPORTS,
        "protein_data_loader": PROTEIN_DATA_LOADER_IMPORTS,
    }
    registry = registry_map.get(component_type)
    if registry is None:
        return False
    return name in registry
