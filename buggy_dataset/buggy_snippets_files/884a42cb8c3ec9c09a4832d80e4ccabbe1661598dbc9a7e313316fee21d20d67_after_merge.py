def init_ndarray(values, index, columns, dtype=None, copy=False):
    # input must be a ndarray, list, Series, index

    if isinstance(values, ABCSeries):
        if columns is None:
            if values.name is not None:
                columns = [values.name]
        if index is None:
            index = values.index
        else:
            values = values.reindex(index)

        # zero len case (GH #2234)
        if not len(values) and columns is not None and len(columns):
            values = np.empty((0, 1), dtype=object)

    # we could have a categorical type passed or coerced to 'category'
    # recast this to an arrays_to_mgr
    if (is_categorical_dtype(getattr(values, 'dtype', None)) or
            is_categorical_dtype(dtype)):

        if not hasattr(values, 'dtype'):
            values = prep_ndarray(values, copy=copy)
            values = values.ravel()
        elif copy:
            values = values.copy()

        index, columns = _get_axes(len(values), 1, index, columns)
        return arrays_to_mgr([values], columns, index, columns,
                             dtype=dtype)
    elif is_extension_array_dtype(values):
        # GH#19157
        if columns is None:
            columns = [0]
        return arrays_to_mgr([values], columns, index, columns,
                             dtype=dtype)

    # by definition an array here
    # the dtypes will be coerced to a single dtype
    values = prep_ndarray(values, copy=copy)

    if dtype is not None:
        if not is_dtype_equal(values.dtype, dtype):
            try:
                values = values.astype(dtype)
            except Exception as orig:
                e = ValueError("failed to cast to '{dtype}' (Exception "
                               "was: {orig})".format(dtype=dtype,
                                                     orig=orig))
                raise_with_traceback(e)

    index, columns = _get_axes(*values.shape, index=index, columns=columns)
    values = values.T

    # if we don't have a dtype specified, then try to convert objects
    # on the entire block; this is to convert if we have datetimelike's
    # embedded in an object type
    if dtype is None and is_object_dtype(values):

        if values.ndim == 2 and values.shape[0] != 1:
            # transpose and separate blocks

            dvals_list = [maybe_infer_to_datetimelike(row) for row in values]
            for n in range(len(dvals_list)):
                if isinstance(dvals_list[n], np.ndarray):
                    dvals_list[n] = dvals_list[n].reshape(1, -1)

            from pandas.core.internals.blocks import make_block

            # TODO: What about re-joining object columns?
            block_values = [make_block(dvals_list[n], placement=[n])
                            for n in range(len(dvals_list))]

        else:
            datelike_vals = maybe_infer_to_datetimelike(values)
            block_values = [datelike_vals]
    else:
        block_values = [values]

    return create_block_manager_from_blocks(block_values, [columns, index])