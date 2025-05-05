"""GPU-specific optimizations for the arithmetic transformer model.

This module provides optimizations for GPU training.
"""

import os
import torch
import warnings

# Check if running on GPU
is_gpu = torch.cuda.is_available()

def apply_optimizations():
    """Apply all necessary optimizations for GPU training."""
    if not is_gpu:
        return
    
    # Enable Tensor Core optimizations
    enable_tensor_cores()
    
    # Set environment variables for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # Print confirmation
    print("Applied GPU optimization settings")

def enable_tensor_cores():
    """Enable Tensor Core optimizations for CUDA devices.
    
    This sets the matmul precision to 'high' to properly utilize Tensor Cores
    on NVIDIA GPUs that support them (like A100, V100, etc).
    """
    if not is_gpu:
        return
        
    # Check if the GPU has Tensor Cores
    device_name = torch.cuda.get_device_name(0)
    tensor_core_gpus = ["A100", "V100", "T4", "RTX", "TITAN", "Quadro", "H100"]
    
    has_tensor_cores = any(gpu_type in device_name for gpu_type in tensor_core_gpus)
    
    if has_tensor_cores:
        # Set matmul precision to 'high' for better performance
        torch.set_float32_matmul_precision('high')
        warnings.warn(f"Enabled Tensor Core optimizations for {device_name}")
