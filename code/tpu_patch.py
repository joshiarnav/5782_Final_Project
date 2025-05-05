"""Patch for PyTorch XLA compatibility issues.

This module provides patches and workarounds for known issues with PyTorch XLA (TPU) backend.
"""

import os
import torch
import warnings

# Check if running on TPU
is_tpu = 'TPU_NAME' in os.environ

def apply_patches():
    """Apply all necessary patches for TPU compatibility."""
    if not is_tpu:
        return
    
    # Apply patch for torch.isin
    patch_torch_isin()
    
    # Print confirmation
    print("Applied TPU compatibility patches")

def patch_torch_isin():
    """Patch torch.isin to work with TPU.
    
    The original implementation uses operations not supported by XLA.
    This patch provides a TPU-compatible version.
    """
    original_isin = torch.isin
    
    def tpu_friendly_isin(elements, test_elements, *, assume_unique=False, invert=False):
        """TPU-friendly implementation of torch.isin."""
        if not is_tpu:
            return original_isin(elements, test_elements, assume_unique=assume_unique, invert=invert)
        
        # Convert to tensor if needed
        if not isinstance(test_elements, torch.Tensor):
            test_elements = torch.tensor(test_elements, device=elements.device)
        
        # Ensure test_elements is at least 1D
        if test_elements.ndim == 0:
            test_elements = test_elements.unsqueeze(0)
        
        # Simple implementation that works on TPU
        result = torch.zeros_like(elements, dtype=torch.bool)
        for val in test_elements:
            result = result | (elements == val)
        
        # Handle invert flag
        if invert:
            result = ~result
            
        return result
    
    # Replace the original function
    torch.isin = tpu_friendly_isin
    warnings.warn("Replaced torch.isin with TPU-compatible version")
