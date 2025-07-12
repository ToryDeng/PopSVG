import sys

from .estimation import TwoStageExpectationMaximizer
from .prepare import preprocess_slices

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

version = __version__


def __getattr__(name):
    if name == "spatial_lag_regression_cpu":
        from .regression_cpu import spatial_lag_regression

        setattr(sys.modules[__name__], name, spatial_lag_regression)
        return spatial_lag_regression
    elif name == "spatial_lag_regression_gpu":
        from .regression_gpu import spatial_lag_regression

        setattr(sys.modules[__name__], name, spatial_lag_regression)
        return spatial_lag_regression
    raise AttributeError(f"Module 'popsvg' has no attribute '{name}'")
