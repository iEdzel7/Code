def lstsq(a, b, rcond="warn"):
    """
    Return the least-squares solution to a linear matrix equation.

    Solves the equation `a x = b` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `a` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `a`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.

    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value of `a`.

        .. versionchanged:: 1.14.0
           If not set, a FutureWarning is given. The previous default
           of ``-1`` will use the machine precision as `rcond` parameter,
           the new default will use the machine precision times `max(M, N)`.
           To silence the warning and use the new default, use ``rcond=None``,
           to keep using the old behavior, use ``rcond=-1``.

    Returns
    -------
    x : {(N,), (N, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    residuals : {(), (1,), (K,)} ndarray
        Sums of residuals; squared Euclidean 2-norm for each column in
        ``b - a*x``.
        If the rank of `a` is < N or M <= N, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : (min(M, N),) ndarray
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    Notes
    -----
    If `b` is a matrix, then all array results are returned as matrices.

    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:

    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])

    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cut the y-axis at, more or less, -1.

    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
    and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:

    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])

    >>> m, c = np.linalg.lstsq(A, y)[0]
    >>> print(m, c)
    1.0 -0.95

    Plot the data along with the fitted line:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o', label='Original data', markersize=10)
    >>> plt.plot(x, m*x + c, 'r', label='Fitted line')
    >>> plt.legend()
    >>> plt.show()

    """
    import math
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    is_1d = b.ndim == 1
    if is_1d:
        b = b[:, newaxis]
    _assertRank2(a, b)
    _assertNoEmpty2d(a, b)  # TODO: relax this constraint
    m  = a.shape[0]
    n  = a.shape[1]
    n_rhs = b.shape[1]
    ldb = max(n, m)
    if m != b.shape[0]:
        raise LinAlgError('Incompatible dimensions')
    t, result_t = _commonType(a, b)
    # Determine default rcond value
    if rcond == "warn":
        # 2017-08-19, 1.14.0
        warnings.warn("`rcond` parameter will change to the default of "
                      "machine precision times ``max(M, N)`` where M and N "
                      "are the input matrix dimensions.\n"
                      "To use the future default and silence this warning "
                      "we advise to pass `rcond=None`, to keep using the old, "
                      "explicitly pass `rcond=-1`.",
                      FutureWarning, stacklevel=2)
        rcond = -1
    if rcond is None:
        rcond = finfo(t).eps * ldb

    result_real_t = _realType(result_t)
    real_t = _linalgRealType(t)
    bstar = zeros((ldb, n_rhs), t)
    bstar[:b.shape[0], :n_rhs] = b.copy()
    a, bstar = _fastCopyAndTranspose(t, a, bstar)
    a, bstar = _to_native_byte_order(a, bstar)
    s = zeros((min(m, n),), real_t)
    # This line:
    #  * is incorrect, according to the LAPACK documentation
    #  * raises a ValueError if min(m,n) == 0
    #  * should not be calculated here anyway, as LAPACK should calculate
    #    `liwork` for us. But that only works if our version of lapack does
    #    not have this bug:
    #      http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00899.html
    #    Lapack_lite does have that bug...
    nlvl = max( 0, int( math.log( float(min(m, n))/2. ) ) + 1 )
    iwork = zeros((3*min(m, n)*nlvl+11*min(m, n),), fortran_int)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgelsd
        lwork = 1
        rwork = zeros((lwork,), real_t)
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, rwork, iwork, 0)
        lwork = int(abs(work[0]))
        rwork = zeros((lwork,), real_t)
        a_real = zeros((m, n), real_t)
        bstar_real = zeros((ldb, n_rhs,), real_t)
        results = lapack_lite.dgelsd(m, n, n_rhs, a_real, m,
                                     bstar_real, ldb, s, rcond,
                                     0, rwork, -1, iwork, 0)
        lrwork = int(rwork[0])
        work = zeros((lwork,), t)
        rwork = zeros((lrwork,), real_t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, lwork, rwork, iwork, 0)
    else:
        lapack_routine = lapack_lite.dgelsd
        lwork = 1
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, -1, iwork, 0)
        lwork = int(work[0])
        work = zeros((lwork,), t)
        results = lapack_routine(m, n, n_rhs, a, m, bstar, ldb, s, rcond,
                                 0, work, lwork, iwork, 0)
    if results['info'] > 0:
        raise LinAlgError('SVD did not converge in Linear Least Squares')
    resids = array([], result_real_t)
    if is_1d:
        x = array(ravel(bstar)[:n], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            if isComplexType(t):
                resids = array([sum(abs(ravel(bstar)[n:])**2)],
                               dtype=result_real_t)
            else:
                resids = array([sum((ravel(bstar)[n:])**2)],
                               dtype=result_real_t)
    else:
        x = array(transpose(bstar)[:n,:], dtype=result_t, copy=True)
        if results['rank'] == n and m > n:
            if isComplexType(t):
                resids = sum(abs(transpose(bstar)[n:,:])**2, axis=0).astype(
                    result_real_t, copy=False)
            else:
                resids = sum((transpose(bstar)[n:,:])**2, axis=0).astype(
                    result_real_t, copy=False)

    st = s[:min(n, m)].astype(result_real_t, copy=True)
    return wrap(x), wrap(resids), results['rank'], st