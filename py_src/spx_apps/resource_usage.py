import logging
import platform
import resource


def get_resource_usage() -> dict:
    """Get resource usage statistics (cross-platform for macOS and Linux).

    Returns
    -------
    dict
        Dictionary with 'peak_memory_mb', 'user_time_sec', 'system_time_sec',
        and 'cpu_time_sec' (total of user + system).

    """
    usage = resource.getrusage(resource.RUSAGE_SELF)

    if platform.system() == 'Darwin':
        # macOS: ru_maxrss is in bytes
        peak_memory_mb = usage.ru_maxrss / 1024 / 1024
    else:
        # Linux: ru_maxrss is in kilobytes
        peak_memory_mb = usage.ru_maxrss / 1024

    return {
        'peak_memory_mb': peak_memory_mb,
        'user_time_sec': usage.ru_utime,
        'system_time_sec': usage.ru_stime,
        'cpu_time_sec': usage.ru_utime + usage.ru_stime,
    }


def log_resource_usage(stage_desc: str = None):
    """Log CPU time and peak memory usage for a processing stage."""
    usage = get_resource_usage()
    if stage_desc is not None:
        logging.info(f"{stage_desc}")
    logging.info(f"CPU time: {usage['cpu_time_sec']:.1f} sec "
                 f"(user: {usage['user_time_sec']:.1f}, "
                 f"system: {usage['system_time_sec']:.1f})")
    logging.info(f"Peak memory: {usage['peak_memory_mb']:.1f} MB")
