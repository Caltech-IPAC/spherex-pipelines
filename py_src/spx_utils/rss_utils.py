__all__ = ["get_current_rss"]

# Standard library imports
import os
import psutil


def get_current_rss(desc):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss  # bytes
    rss_mib = rss / (1024**2)  # convert to MiB
    info_msg = f"{desc} RSS (PID {process}): {rss_mib:.5f} MiB"
    return info_msg
