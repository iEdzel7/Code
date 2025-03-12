def infer_dtype_from_scalar(val, pandas_dtype: bool = False) -> Tuple[DtypeObj, Any]:
    """
    Interpret the dtype from a scalar.

    Parameters
    ----------
    pandas_dtype : bool, default False
        whether to infer dtype including pandas extension types.
        If False, scalar belongs to pandas extension types is inferred as
        object
    """
    dtype: DtypeObj = np.dtype(object)

    # a 1-element ndarray
    if isinstance(val, np.ndarray):
        msg = "invalid ndarray passed to infer_dtype_from_scalar"
        if val.ndim != 0:
            raise ValueError(msg)

        dtype = val.dtype
        val = val.item()

    elif isinstance(val, str):

        # If we create an empty array using a string to infer
        # the dtype, NumPy will only allocate one character per entry
        # so this is kind of bad. Alternately we could use np.repeat
        # instead of np.empty (but then you still don't want things
        # coming out as np.str_!

        dtype = np.dtype(object)

    elif isinstance(val, (np.datetime64, datetime)):
        val = tslibs.Timestamp(val)
        if val is tslibs.NaT or val.tz is None:
            dtype = np.dtype("M8[ns]")
        else:
            if pandas_dtype:
                dtype = DatetimeTZDtype(unit="ns", tz=val.tz)
            else:
                # return datetimetz as object
                return np.dtype(object), val
        val = val.value

    elif isinstance(val, (np.timedelta64, timedelta)):
        val = tslibs.Timedelta(val).value
        dtype = np.dtype("m8[ns]")

    elif is_bool(val):
        dtype = np.dtype(np.bool_)

    elif is_integer(val):
        if isinstance(val, np.integer):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.int64)

        try:
            np.array(val, dtype=dtype)
        except OverflowError:
            dtype = np.array(val).dtype

    elif is_float(val):
        if isinstance(val, np.floating):
            dtype = np.dtype(type(val))
        else:
            dtype = np.dtype(np.float64)

    elif is_complex(val):
        dtype = np.dtype(np.complex_)

    elif pandas_dtype:
        if lib.is_period(val):
            dtype = PeriodDtype(freq=val.freq)
        elif lib.is_interval(val):
            subtype = infer_dtype_from_scalar(val.left, pandas_dtype=True)[0]
            dtype = IntervalDtype(subtype=subtype)

    return dtype, val