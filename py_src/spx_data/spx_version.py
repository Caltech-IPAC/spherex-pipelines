__all__ = ["SSDC_PIPELINES_VERSION"]
import importlib.metadata

try:
    SSDC_PIPELINES_VERSION = importlib.metadata.version("spherex_pipelines")
except importlib.metadata.PackageNotFoundError:
    SSDC_PIPELINES_VERSION = "unknown"
