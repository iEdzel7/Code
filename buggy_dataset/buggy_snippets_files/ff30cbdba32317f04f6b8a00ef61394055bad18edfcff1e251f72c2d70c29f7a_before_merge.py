def byte_bounds(a):
    """
    Returns pointers to the end-points of an array.

    Parameters
    ----------
    a : ndarray
        Input array. It must conform to the Python-side of the array interface.

    Returns
    -------
    (low, high) : tuple of 2 integers
        The first integer is the first byte of the array, the second integer is
        just past the last byte of the array.  If `a` is not contiguous it
        will not use every byte between the (`low`, `high`) values.

    Examples
    --------
    >>> I = np.eye(2, dtype='f'); I.dtype
    dtype('float32')
    >>> low, high = np.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True
    >>> I = np.eye(2, dtype='G'); I.dtype
    dtype('complex192')
    >>> low, high = np.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True

    """
    ai = a.__array_interface__
    a_data = ai['data'][0]
    astrides = ai['strides']
    ashape = ai['shape']
    bytes_a = int(ai['typestr'][2:])

    a_low = a_high = a_data
    if astrides is None: # contiguous case
        a_high += a.size * bytes_a
    else:
        for shape, stride in zip(ashape, astrides):
            if stride < 0:
                a_low += (shape-1)*stride
            else:
                a_high += (shape-1)*stride
        a_high += bytes_a
    return a_low, a_high