def get_akshare():
    """
    Get the akshare module.
    Because akshare is frequently updated, we need to check if it is installed correctly.
    """
    import akshare as ak

    if ak is None:
        raise ImportError("Failed to import the akshare module.")

    return ak
