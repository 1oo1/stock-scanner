import concurrent.futures

_executor = None


def get_thread_pool_executor():
    """Creates and returns a ThreadPoolExecutor instance."""
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="stock_updater"
        )
    return _executor
