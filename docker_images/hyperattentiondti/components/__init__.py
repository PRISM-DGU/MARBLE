"""HyperAttentionDTI Components Package

Provides lazy-loading registry for drug encoder, protein encoder, and decoder components.
"""

from components.registry import (
    get_drug_encoder,
    get_protein_encoder,
    get_decoder,
    get_drug_data_loader,
    get_protein_data_loader,
    list_available_components,
    is_registered,
)

__all__ = [
    "get_drug_encoder",
    "get_protein_encoder",
    "get_decoder",
    "get_drug_data_loader",
    "get_protein_data_loader",
    "list_available_components",
    "is_registered",
]
