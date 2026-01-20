"""Runtime Validator - Dynamic Error Detection and Handling

This module provides runtime validation and automatic error correction for
common issues that may arise from LLM-generated code.

Key features:
1. Encoder output validation (tuple vs tensor)
2. GATConv dimension validation and auto-correction
3. Edge index validation
4. Tensor shape validation
5. Gradient flow checking
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
from functools import wraps
from typing import Union, Tuple, Any, Optional


class RuntimeValidator:
    """Central validator for runtime error detection and correction"""

    @staticmethod
    def validate_encoder_output(output: Any, expected_dim: int = None) -> torch.Tensor:
        """
        Validate and normalize encoder output.

        Common LLM mistakes:
        - Returning tuple (embedding, reconstruction, aux1, aux2) instead of just embedding
        - Returning dict instead of tensor

        Args:
            output: Raw encoder output (could be tensor, tuple, or dict)
            expected_dim: Expected output dimension (optional)

        Returns:
            torch.Tensor: Validated embedding tensor
        """
        # Case 1: Already a tensor
        if isinstance(output, torch.Tensor):
            return output

        # Case 2: Tuple - take the first element (usually the main embedding)
        if isinstance(output, (tuple, list)):
            warnings.warn(
                f"Encoder returned {type(output).__name__} with {len(output)} elements. "
                f"Using first element as embedding. Consider fixing the encoder to return only the embedding.",
                RuntimeWarning
            )
            first_elem = output[0]
            if isinstance(first_elem, torch.Tensor):
                return first_elem
            else:
                raise ValueError(f"First element of encoder output is not a tensor: {type(first_elem)}")

        # Case 3: Dict - look for common keys
        if isinstance(output, dict):
            for key in ['embedding', 'z', 'latent', 'hidden', 'output', 'h']:
                if key in output and isinstance(output[key], torch.Tensor):
                    warnings.warn(
                        f"Encoder returned dict. Using '{key}' as embedding.",
                        RuntimeWarning
                    )
                    return output[key]
            raise ValueError(f"Could not find embedding tensor in encoder output dict. Keys: {list(output.keys())}")

        raise TypeError(f"Unsupported encoder output type: {type(output)}")

    @staticmethod
    def validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Validate and fix edge_index tensor.

        Common issues:
        - NaN values from unmapped cell names
        - Out-of-bounds node indices
        - Wrong dtype

        Args:
            edge_index: Edge index tensor of shape (2, num_edges)
            num_nodes: Number of nodes in the graph

        Returns:
            torch.Tensor: Validated edge_index
        """
        if edge_index is None:
            return None

        # Ensure correct dtype
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        # Check for NaN (shouldn't happen with long dtype, but check original)
        if torch.isnan(edge_index.float()).any():
            warnings.warn("Edge index contains NaN values. Removing invalid edges.", RuntimeWarning)
            valid_mask = ~torch.isnan(edge_index.float()).any(dim=0)
            edge_index = edge_index[:, valid_mask]

        # Check bounds
        max_idx = edge_index.max().item()
        min_idx = edge_index.min().item()

        if min_idx < 0:
            warnings.warn(f"Edge index contains negative indices (min={min_idx}). Removing invalid edges.", RuntimeWarning)
            valid_mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
            edge_index = edge_index[:, valid_mask]

        if max_idx >= num_nodes:
            warnings.warn(
                f"Edge index contains out-of-bounds indices (max={max_idx}, num_nodes={num_nodes}). "
                f"Removing invalid edges.",
                RuntimeWarning
            )
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_mask]

        return edge_index

    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...],
                             name: str = "tensor") -> bool:
        """
        Validate tensor shape and provide diagnostic info.

        Args:
            tensor: Tensor to validate
            expected_shape: Expected shape (-1 for any dimension)
            name: Name for error messages

        Returns:
            bool: True if valid
        """
        if tensor is None:
            return True

        actual_shape = tensor.shape

        if len(actual_shape) != len(expected_shape):
            warnings.warn(
                f"{name} has wrong number of dimensions. "
                f"Expected {len(expected_shape)}, got {len(actual_shape)}. "
                f"Shape: {actual_shape}",
                RuntimeWarning
            )
            return False

        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected != -1 and actual != expected:
                warnings.warn(
                    f"{name} dimension {i} mismatch. "
                    f"Expected {expected}, got {actual}. "
                    f"Full shape: {actual_shape}",
                    RuntimeWarning
                )
                return False

        return True

    @staticmethod
    def fix_gatconv_dimensions(module: nn.Module) -> bool:
        """
        Fix GATConv dimension issues dynamically.

        Common LLM mistake:
        - lin_src shape is [in_channels, out_channels] instead of [in_channels, heads * out_channels]

        Args:
            module: Module to check and fix

        Returns:
            bool: True if fix was applied
        """
        fixed = False

        for name, child in module.named_modules():
            # Check if this looks like a GATConv
            if hasattr(child, 'lin_src') and hasattr(child, 'heads') and hasattr(child, 'out_channels'):
                lin_src = child.lin_src
                heads = child.heads
                out_channels = child.out_channels
                in_channels = child.in_channels if hasattr(child, 'in_channels') else lin_src.shape[0]

                expected_shape = (in_channels, heads * out_channels)
                actual_shape = tuple(lin_src.shape)

                if actual_shape != expected_shape:
                    # Check if it's the common mistake
                    if actual_shape == (in_channels, out_channels):
                        warnings.warn(
                            f"Fixing GATConv '{name}' lin_src dimension: "
                            f"{actual_shape} -> {expected_shape}",
                            RuntimeWarning
                        )

                        # Create new parameter with correct shape
                        new_lin_src = nn.Parameter(torch.zeros(expected_shape,
                                                              dtype=lin_src.dtype,
                                                              device=lin_src.device))
                        nn.init.xavier_normal_(new_lin_src.data, gain=1.414)
                        child.lin_src = new_lin_src
                        child.lin_dst = new_lin_src  # Usually shared

                        fixed = True

        return fixed


class ValidatedModel(nn.Module):
    """Wrapper that adds validation to any model"""

    def __init__(self, model: nn.Module, validate_encoder: bool = True,
                 fix_gatconv: bool = True):
        super().__init__()
        self.model = model
        self.validate_encoder = validate_encoder
        self._initialized = False

        # Fix GATConv dimensions if needed
        if fix_gatconv:
            RuntimeValidator.fix_gatconv_dimensions(model)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None,
                adata=None, **kwargs):
        # Validate edge_index
        if edge_index is not None:
            edge_index = RuntimeValidator.validate_edge_index(edge_index, x.shape[0])

        # Run model
        output = self.model(x=x, edge_index=edge_index, adata=adata, **kwargs)

        return output

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def validate_forward(func):
    """Decorator to validate forward pass inputs and outputs"""
    @wraps(func)
    def wrapper(self, x: torch.Tensor, edge_index: torch.Tensor = None, **kwargs):
        # Validate inputs
        if edge_index is not None and x is not None:
            edge_index = RuntimeValidator.validate_edge_index(edge_index, x.shape[0])

        # Run forward
        output = func(self, x=x, edge_index=edge_index, **kwargs)

        return output
    return wrapper


def normalize_encoder_output(func):
    """Decorator to normalize encoder output to single tensor"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        output = func(self, *args, **kwargs)
        return RuntimeValidator.validate_encoder_output(output)
    return wrapper


class ErrorDiagnostics:
    """Provides detailed error diagnostics for debugging"""

    @staticmethod
    def diagnose_cuda_index_error(x: torch.Tensor, edge_index: torch.Tensor,
                                  model: nn.Module) -> str:
        """
        Diagnose CUDA index errors.

        Args:
            x: Input features
            edge_index: Graph connectivity
            model: Model that caused the error

        Returns:
            str: Diagnostic message
        """
        diagnostics = []
        diagnostics.append("=" * 60)
        diagnostics.append("CUDA INDEX ERROR DIAGNOSTICS")
        diagnostics.append("=" * 60)

        # Check input shapes
        diagnostics.append(f"\n[Input Shapes]")
        diagnostics.append(f"  x.shape: {x.shape}")
        diagnostics.append(f"  x.device: {x.device}")
        diagnostics.append(f"  x.dtype: {x.dtype}")

        if edge_index is not None:
            diagnostics.append(f"  edge_index.shape: {edge_index.shape}")
            diagnostics.append(f"  edge_index.device: {edge_index.device}")
            diagnostics.append(f"  edge_index.dtype: {edge_index.dtype}")
            diagnostics.append(f"  edge_index min: {edge_index.min().item()}")
            diagnostics.append(f"  edge_index max: {edge_index.max().item()}")
            diagnostics.append(f"  num_nodes: {x.shape[0]}")

            if edge_index.max().item() >= x.shape[0]:
                diagnostics.append(f"\n  [ERROR] edge_index max ({edge_index.max().item()}) >= num_nodes ({x.shape[0]})")

        # Check model parameters
        diagnostics.append(f"\n[Model Structure]")
        for name, param in model.named_parameters():
            if 'lin' in name.lower():
                diagnostics.append(f"  {name}: {param.shape}")

        # Check for GATConv issues
        diagnostics.append(f"\n[GATConv Check]")
        for name, module in model.named_modules():
            if hasattr(module, 'heads') and hasattr(module, 'out_channels'):
                in_ch = getattr(module, 'in_channels', 'N/A')
                out_ch = module.out_channels
                heads = module.heads
                diagnostics.append(f"  {name}: in={in_ch}, out={out_ch}, heads={heads}")

                if hasattr(module, 'lin_src'):
                    lin_shape = module.lin_src.shape
                    expected = (in_ch, heads * out_ch) if in_ch != 'N/A' else 'N/A'
                    diagnostics.append(f"    lin_src.shape: {lin_shape} (expected: {expected})")
                    if expected != 'N/A' and lin_shape != expected:
                        diagnostics.append(f"    [ERROR] Dimension mismatch!")

        return "\n".join(diagnostics)

    @staticmethod
    def diagnose_tensor_mismatch(tensors: dict) -> str:
        """
        Diagnose tensor shape/device mismatches.

        Args:
            tensors: Dict of tensor_name -> tensor

        Returns:
            str: Diagnostic message
        """
        diagnostics = []
        diagnostics.append("=" * 60)
        diagnostics.append("TENSOR MISMATCH DIAGNOSTICS")
        diagnostics.append("=" * 60)

        devices = set()
        dtypes = set()

        for name, tensor in tensors.items():
            if tensor is not None:
                diagnostics.append(f"\n{name}:")
                diagnostics.append(f"  shape: {tensor.shape}")
                diagnostics.append(f"  device: {tensor.device}")
                diagnostics.append(f"  dtype: {tensor.dtype}")
                devices.add(str(tensor.device))
                dtypes.add(str(tensor.dtype))

        if len(devices) > 1:
            diagnostics.append(f"\n[WARNING] Multiple devices detected: {devices}")
        if len(dtypes) > 1:
            diagnostics.append(f"\n[WARNING] Multiple dtypes detected: {dtypes}")

        return "\n".join(diagnostics)
