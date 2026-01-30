from ._loader import get_core_safely

PointerFlowBuffer, as_numpy_float32 = get_core_safely()

__all__ = [
    "PointerFlowBuffer",
    "as_numpy_float32",
]