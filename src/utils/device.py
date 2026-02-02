"""
Device detection and configuration utility.
Automatically detects the best available compute device (MPS/CUDA/CPU).
"""

import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types."""

    MPS = "mps"  # Apple Silicon
    CUDA = "cuda"  # NVIDIA GPU
    CPU = "cpu"  # Fallback


@dataclass
class DeviceConfig:
    """Device configuration for model loading."""

    device: str
    device_type: DeviceType
    dtype: torch.dtype
    attn_implementation: str

    def __str__(self) -> str:
        return f"DeviceConfig(device={self.device}, dtype={self.dtype}, attn={self.attn_implementation})"


def detect_device(force_device: Optional[str] = None) -> DeviceConfig:
    """
    Detect the best available compute device.

    Args:
        force_device: Override device detection with specific device (mps/cuda/cpu)

    Returns:
        DeviceConfig with optimal settings for the detected device
    """
    # Check environment variable override
    env_device = os.environ.get("DEVICE", "").lower()
    if env_device:
        force_device = env_device

    if force_device:
        force_device = force_device.lower()
        if force_device == "mps":
            return _get_mps_config()
        elif force_device == "cuda":
            return _get_cuda_config()
        else:
            return _get_cpu_config()

    # Auto-detect
    if torch.backends.mps.is_available():
        logger.info("Detected Apple Silicon - using MPS")
        return _get_mps_config()
    elif torch.cuda.is_available():
        logger.info(f"Detected CUDA GPU - {torch.cuda.get_device_name(0)}")
        return _get_cuda_config()
    else:
        logger.info("No GPU detected - using CPU")
        return _get_cpu_config()


def _get_mps_config() -> DeviceConfig:
    """Configuration for Apple Silicon MPS."""
    return DeviceConfig(
        device="mps",
        device_type=DeviceType.MPS,
        # IMPORTANT: float32 required for voice cloning on MPS
        dtype=torch.float32,
        # SDPA instead of FlashAttention (not available on MPS)
        attn_implementation="sdpa",
    )


def _get_cuda_config() -> DeviceConfig:
    """Configuration for NVIDIA CUDA GPU."""
    return DeviceConfig(
        device="cuda:0",
        device_type=DeviceType.CUDA,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


def _get_cpu_config() -> DeviceConfig:
    """Configuration for CPU fallback."""
    return DeviceConfig(
        device="cpu",
        device_type=DeviceType.CPU,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )


def get_device_info() -> dict:
    """Get detailed device information for debugging."""
    info = {
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory

    return info


if __name__ == "__main__":
    # Test device detection
    logging.basicConfig(level=logging.INFO)
    print("Device Info:", get_device_info())
    print("Detected Config:", detect_device())
