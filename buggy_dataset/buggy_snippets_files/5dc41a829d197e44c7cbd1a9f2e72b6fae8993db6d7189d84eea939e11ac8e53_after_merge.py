def _coerce_to_type(x):
    """
    if the passed data is of datetime/timedelta type,
    this method converts it to numeric so that cut method can
    handle it
    """
    dtype = None

    if is_timedelta64_dtype(x):
        x = to_timedelta(x)
        dtype = np.timedelta64
    elif is_datetime64_dtype(x):
        x = to_datetime(x)
        dtype = np.datetime64

    if dtype is not None:
        # GH 19768: force NaT to NaN during integer conversion
        x = np.where(x.notna(), x.view(np.int64), np.nan)

    return x, dtype