def _isna_old(obj):
    """Detect missing values. Treat None, NaN, INF, -INF as null.

    Parameters
    ----------
    arr: ndarray or object value

    Returns
    -------
    boolean ndarray or boolean
    """
    if is_scalar(obj):
        return libmissing.checknull_old(obj)
    # hack (for now) because MI registers as ndarray
    elif isinstance(obj, ABCMultiIndex):
        raise NotImplementedError("isna is not defined for MultiIndex")
    elif isinstance(obj, (ABCSeries, np.ndarray, ABCIndexClass)):
        return _isna_ndarraylike_old(obj)
    elif isinstance(obj, ABCGeneric):
        return obj._constructor(obj._data.isna(func=_isna_old))
    elif isinstance(obj, list):
        return _isna_ndarraylike_old(np.asarray(obj, dtype=object))
    elif hasattr(obj, "__array__"):
        return _isna_ndarraylike_old(np.asarray(obj))
    else:
        return obj is None