import os
import sys
from pathlib import Path


def setup_cuda_environment():
    """
    Configures CUDA DLL paths on Windows to avoid 'DLL load failed' errors.
    Does nothing on Linux/macOS (they use LD_LIBRARY_PATH or rpath instead).
    """
    # Only relevant on Windows with Python 3.8+
    if sys.platform != "win32" or sys.version_info < (3, 8):
        return

    # Try to get CUDA_PATH from environment variable
    cuda_path = os.environ.get("CUDA_PATH")

    # If not set, find the latest CUDA installation in the default path
    if not cuda_path:
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.exists():
            # Sort by name descending to get the newest version first
            versions = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
            if versions:
                cuda_path = str(versions[0])

    # Add the bin directory to DLL search path if it exists
    if cuda_path:
        bin_dir = Path(cuda_path) / "bin"
        if bin_dir.exists():
            os.add_dll_directory(str(bin_dir))


def get_core_safely():
    """
    Safely imports the native module after setting up CUDA paths.
    Returns PointerFlowBuffer and as_numpy_float32 or raises a clear error.
    """
    # Ensure DLL paths are ready before import (Windows only)
    setup_cuda_environment()

    try:
        from . import _core
        return _core.PointerFlowBuffer, _core.as_numpy_float32
    except ImportError as e:
        _raise_pretty_error(e)


def _raise_pretty_error(e):
    """
    Raises a user-friendly ImportError with platform-specific advice.
    """
    if sys.platform == "win32":
        msg = (
            "Failed to load _core.pyd on Windows. "
            "Check that CUDA bin directory is in PATH or NVIDIA drivers are installed. "
            "Manual fix: os.add_dll_directory(r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin')"
        )
    elif sys.platform.startswith("linux"):
        msg = (
            "Failed to load _core.so on Linux. "
            "Check LD_LIBRARY_PATH includes /usr/local/cuda/lib64 or your CUDA toolkit path. "
            "Or rebuild with rpath enabled in CMake."
        )
    else:
        msg = "Failed to load the native module. Check CUDA installation."

    raise ImportError(msg) from e