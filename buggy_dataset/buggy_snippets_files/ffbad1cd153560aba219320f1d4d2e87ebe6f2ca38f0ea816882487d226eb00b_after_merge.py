def masked_where(condition, a, copy=True):
    """
    Mask an array where a condition is met.

    Return `a` as an array masked where `condition` is True.
    Any masked values of `a` or `condition` are also masked in the output.

    Parameters
    ----------
    condition : array_like
        Masking condition.  When `condition` tests floating point values for
        equality, consider using ``masked_values`` instead.
    a : array_like
        Array to mask.
    copy : bool
        If True (default) make a copy of `a` in the result.  If False modify
        `a` in place and return a view.

    Returns
    -------
    result : MaskedArray
        The result of masking `a` where `condition` is True.

    See Also
    --------
    masked_values : Mask using floating point equality.
    masked_equal : Mask where equal to a given value.
    masked_not_equal : Mask where `not` equal to a given value.
    masked_less_equal : Mask where less than or equal to a given value.
    masked_greater_equal : Mask where greater than or equal to a given value.
    masked_less : Mask where less than a given value.
    masked_greater : Mask where greater than a given value.
    masked_inside : Mask inside a given interval.
    masked_outside : Mask outside a given interval.
    masked_invalid : Mask invalid values (NaNs or infs).

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_where(a <= 2, a)
    masked_array(data = [-- -- -- 3],
          mask = [ True  True  True False],
          fill_value=999999)

    Mask array `b` conditional on `a`.

    >>> b = ['a', 'b', 'c', 'd']
    >>> ma.masked_where(a == 2, b)
    masked_array(data = [a b -- d],
          mask = [False False  True False],
          fill_value=N/A)

    Effect of the `copy` argument.

    >>> c = ma.masked_where(a <= 2, a)
    >>> c
    masked_array(data = [-- -- -- 3],
          mask = [ True  True  True False],
          fill_value=999999)
    >>> c[0] = 99
    >>> c
    masked_array(data = [99 -- -- 3],
          mask = [False  True  True False],
          fill_value=999999)
    >>> a
    array([0, 1, 2, 3])
    >>> c = ma.masked_where(a <= 2, a, copy=False)
    >>> c[0] = 99
    >>> c
    masked_array(data = [99 -- -- 3],
          mask = [False  True  True False],
          fill_value=999999)
    >>> a
    array([99,  1,  2,  3])

    When `condition` or `a` contain masked values.

    >>> a = np.arange(4)
    >>> a = ma.masked_where(a == 2, a)
    >>> a
    masked_array(data = [0 1 -- 3],
          mask = [False False  True False],
          fill_value=999999)
    >>> b = np.arange(4)
    >>> b = ma.masked_where(b == 0, b)
    >>> b
    masked_array(data = [-- 1 2 3],
          mask = [ True False False False],
          fill_value=999999)
    >>> ma.masked_where(a == 3, b)
    masked_array(data = [-- 1 -- --],
          mask = [ True False  True  True],
          fill_value=999999)

    """
    # Make sure that condition is a valid standard-type mask.
    cond = make_mask(condition)
    a = np.array(a, copy=copy, subok=True)

    (cshape, ashape) = (cond.shape, a.shape)
    if cshape and cshape != ashape:
        raise IndexError("Inconsistant shape between the condition and the input"
                         " (got %s and %s)" % (cshape, ashape))
    if hasattr(a, '_mask'):
        cond = mask_or(cond, a._mask)
        cls = type(a)
    else:
        cls = MaskedArray
    result = a.view(cls)
    # Assign to *.mask so that structured masks are handled correctly.
    result.mask = cond
    return result