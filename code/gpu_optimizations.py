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
    
    # Configure mixed precision settings
    configure_mixed_precision()
    
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

def configure_mixed_precision():
    """Configure mixed precision settings for optimal performance.
    
    Mixed precision uses FP16 for most operations while maintaining FP32
    for operations that need higher precision, providing significant speedup.
    """
    if not is_gpu:
        return
        
    # Check if the GPU supports mixed precision well
    device_name = torch.cuda.get_device_name(0)
    amp_friendly_gpus = ["A100", "V100", "T4", "RTX", "TITAN", "Quadro", "H100", "A10"]
    
    supports_amp = any(gpu_type in device_name for gpu_type in amp_friendly_gpus)
    
    if supports_amp:
        # Only enable TF32 which is more stable than FP16 mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
        torch.backends.cudnn.allow_tf32 = True        # Allow TF32 on cudnn
        
        # We'll keep using 32-bit precision in the trainer for stability
        warnings.warn(f"Configured TF32 settings for {device_name} (keeping 32-bit precision for stability)")
