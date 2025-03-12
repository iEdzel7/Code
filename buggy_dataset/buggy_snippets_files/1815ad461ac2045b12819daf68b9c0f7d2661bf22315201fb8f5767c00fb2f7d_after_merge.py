def cut(x, bins, right=True, labels=None, retbins=False, precision=3,
        include_lowest=False):
    """
    Return indices of half-open bins to which each value of `x` belongs.

    Parameters
    ----------
    x : array-like
        Input array to be binned. It has to be 1-dimensional.
    bins : int or sequence of scalars
        If `bins` is an int, it defines the number of equal-width bins in the
        range of `x`. However, in this case, the range of `x` is extended
        by .1% on each side to include the min or max values of `x`. If
        `bins` is a sequence it defines the bin edges allowing for
        non-uniform bin width. No extension of the range of `x` is done in
        this case.
    right : bool, optional
        Indicates whether the bins include the rightmost edge or not. If
        right == True (the default), then the bins [1,2,3,4] indicate
        (1,2], (2,3], (3,4].
    labels : array or boolean, default None
        Used as labels for the resulting bins. Must be of the same length as
        the resulting bins. If False, return only integer indicators of the
        bins.
    retbins : bool, optional
        Whether to return the bins or not. Can be useful if bins is given
        as a scalar.
    precision : int
        The precision at which to store and display the bins labels
    include_lowest : bool
        Whether the first interval should be left-inclusive or not.

    Returns
    -------
    out : Categorical or Series or array of integers if labels is False
        The return type (Categorical or Series) depends on the input: a Series
        of type category if input is a Series else Categorical. Bins are
        represented as categories when categorical data is returned.
    bins : ndarray of floats
        Returned only if `retbins` is True.

    Notes
    -----
    The `cut` function can be useful for going from a continuous variable to
    a categorical variable. For example, `cut` could convert ages to groups
    of age ranges.

    Any NA values will be NA in the result.  Out of bounds values will be NA in
    the resulting Categorical object


    Examples
    --------
    >>> pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3, retbins=True)
    ([(0.191, 3.367], (0.191, 3.367], (0.191, 3.367], (3.367, 6.533],
      (6.533, 9.7], (0.191, 3.367]]
    Categories (3, object): [(0.191, 3.367] < (3.367, 6.533] < (6.533, 9.7]],
    array([ 0.1905    ,  3.36666667,  6.53333333,  9.7       ]))
    >>> pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3,
               labels=["good","medium","bad"])
    [good, good, good, medium, bad, good]
    Categories (3, object): [good < medium < bad]
    >>> pd.cut(np.ones(5), 4, labels=False)
    array([1, 1, 1, 1, 1], dtype=int64)
    """
    # NOTE: this binning code is changed a bit from histogram for var(x) == 0

    # for handling the cut for datetime and timedelta objects
    x_is_series, series_index, name, x = _preprocess_for_cut(x)
    x, dtype = _coerce_to_type(x)

    if not np.iterable(bins):
        if is_scalar(bins) and bins < 1:
            raise ValueError("`bins` should be a positive integer.")

        sz = x.size

        if sz == 0:
            raise ValueError('Cannot cut empty array')
            # handle empty arrays. Can't determine range, so use 0-1.
            # rng = (0, 1)
        else:
            rng = (nanops.nanmin(x), nanops.nanmax(x))
        mn, mx = [mi + 0.0 for mi in rng]

        if mn == mx:  # adjust end points before binning
            mn -= .001 * abs(mn) if mn != 0 else .001
            mx += .001 * abs(mx) if mx != 0 else .001
            bins = np.linspace(mn, mx, bins + 1, endpoint=True)
        else:  # adjust end points after binning
            bins = np.linspace(mn, mx, bins + 1, endpoint=True)
            adj = (mx - mn) * 0.001  # 0.1% of the range
            if right:
                bins[0] -= adj
            else:
                bins[-1] += adj

    else:
        bins = np.asarray(bins)
        bins = _convert_bin_to_numeric_type(bins)
        if (np.diff(bins) < 0).any():
            raise ValueError('bins must increase monotonically.')

    fac, bins = _bins_to_cuts(x, bins, right=right, labels=labels,
                              precision=precision,
                              include_lowest=include_lowest, dtype=dtype)

    return _postprocess_for_cut(fac, bins, retbins, x_is_series,
                                series_index, name)