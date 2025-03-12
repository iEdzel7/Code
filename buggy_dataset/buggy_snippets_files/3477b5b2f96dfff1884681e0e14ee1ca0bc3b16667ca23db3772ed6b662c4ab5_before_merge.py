def sanitize_array(
    data,
    index: Optional["Index"],
    dtype: Optional[DtypeObj] = None,
    copy: bool = False,
    raise_cast_failure: bool = False,
) -> ArrayLike:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.
    """

    if isinstance(data, ma.MaskedArray):
        mask = ma.getmaskarray(data)
        if mask.any():
            data, fill_value = maybe_upcast(data, copy=True)
            data.soften_mask()  # set hardmask False if it was True
            data[mask] = fill_value
        else:
            data = data.copy()

    # extract ndarray or ExtensionArray, ensure we have no PandasArray
    data = extract_array(data, extract_numpy=True)

    # GH#846
    if isinstance(data, np.ndarray):

        if dtype is not None and is_float_dtype(data.dtype) and is_integer_dtype(dtype):
            # possibility of nan -> garbage
            try:
                subarr = _try_cast(data, dtype, copy, True)
            except ValueError:
                if copy:
                    subarr = data.copy()
                else:
                    subarr = np.array(data, copy=False)
        else:
            # we will try to copy be-definition here
            subarr = _try_cast(data, dtype, copy, raise_cast_failure)

    elif isinstance(data, ABCExtensionArray):
        # it is already ensured above this is not a PandasArray
        subarr = data

        if dtype is not None:
            subarr = subarr.astype(dtype, copy=copy)
        elif copy:
            subarr = subarr.copy()
        return subarr

    elif isinstance(data, (list, tuple)) and len(data) > 0:
        if dtype is not None:
            subarr = _try_cast(data, dtype, copy, raise_cast_failure)
        else:
            subarr = maybe_convert_platform(data)

        subarr = maybe_cast_to_datetime(subarr, dtype)

    elif isinstance(data, range):
        # GH#16804
        arr = np.arange(data.start, data.stop, data.step, dtype="int64")
        subarr = _try_cast(arr, dtype, copy, raise_cast_failure)
    elif isinstance(data, abc.Set):
        raise TypeError("Set type is unordered")
    else:
        subarr = _try_cast(data, dtype, copy, raise_cast_failure)

    # scalar like, GH
    if getattr(subarr, "ndim", 0) == 0:
        if isinstance(data, list):  # pragma: no cover
            subarr = np.array(data, dtype=object)
        elif index is not None:
            value = data

            # figure out the dtype from the value (upcast if necessary)
            if dtype is None:
                dtype, value = infer_dtype_from_scalar(value)
            else:
                # need to possibly convert the value here
                value = maybe_cast_to_datetime(value, dtype)

            subarr = construct_1d_arraylike_from_scalar(value, len(index), dtype)

        else:
            return subarr.item()

    # the result that we want
    elif subarr.ndim == 1:
        if index is not None:

            # a 1-element ndarray
            if len(subarr) != len(index) and len(subarr) == 1:
                subarr = construct_1d_arraylike_from_scalar(
                    subarr[0], len(index), subarr.dtype
                )

    elif subarr.ndim > 1:
        if isinstance(data, np.ndarray):
            raise Exception("Data must be 1-dimensional")
        else:
            subarr = com.asarray_tuplesafe(data, dtype=dtype)

    if not (is_extension_array_dtype(subarr.dtype) or is_extension_array_dtype(dtype)):
        # This is to prevent mixed-type Series getting all casted to
        # NumPy string type, e.g. NaN --> '-1#IND'.
        if issubclass(subarr.dtype.type, str):
            # GH#16605
            # If not empty convert the data to dtype
            # GH#19853: If data is a scalar, subarr has already the result
            if not lib.is_scalar(data):
                if not np.all(isna(data)):
                    data = np.array(data, dtype=dtype, copy=False)
                subarr = np.array(data, dtype=object, copy=copy)

        if is_object_dtype(subarr.dtype) and not is_object_dtype(dtype):
            inferred = lib.infer_dtype(subarr, skipna=False)
            if inferred in {"interval", "period"}:
                subarr = array(subarr)

    return subarr