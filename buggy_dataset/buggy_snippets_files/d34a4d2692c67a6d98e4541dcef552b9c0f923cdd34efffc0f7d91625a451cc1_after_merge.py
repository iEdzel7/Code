def pairplot(
    data, *,
    hue=None, hue_order=None, palette=None,
    vars=None, x_vars=None, y_vars=None,
    kind="scatter", diag_kind="auto", markers=None,
    height=2.5, aspect=1, corner=False, dropna=False,
    plot_kws=None, diag_kws=None, grid_kws=None, size=None,
):
    """Plot pairwise relationships in a dataset.

    By default, this function will create a grid of Axes such that each numeric
    variable in ``data`` will by shared across the y-axes across a single row and
    the x-axes across a single column. The diagonal plots are treated
    differently: a univariate distribution plot is drawn to show the marginal
    distribution of the data in each column.

    It is also possible to show a subset of variables or plot different
    variables on the rows and columns.

    This is a high-level interface for :class:`PairGrid` that is intended to
    make it easy to draw a few common styles. You should use :class:`PairGrid`
    directly if you need more flexibility.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
    hue : name of variable in ``data``
        Variable in ``data`` to map plot aspects to different colors.
    hue_order : list of strings
        Order for the levels of the hue variable in the palette
    palette : dict or seaborn color palette
        Set of colors for mapping the ``hue`` variable. If a dict, keys
        should be values  in the ``hue`` variable.
    vars : list of variable names
        Variables within ``data`` to use, otherwise use every column with
        a numeric datatype.
    {x, y}_vars : lists of variable names
        Variables within ``data`` to use separately for the rows and
        columns of the figure; i.e. to make a non-square plot.
    kind : {'scatter', 'kde', 'hist', 'reg'}
        Kind of plot to make.
    diag_kind : {'auto', 'hist', 'kde', None}
        Kind of plot for the diagonal subplots. If 'auto', choose based on
        whether or not ``hue`` is used.
    markers : single matplotlib marker code or list
        Either the marker to use for all scatterplot points or a list of markers
        with a length the same as the number of levels in the hue variable so that
        differently colored points will also have different scatterplot
        markers.
    height : scalar
        Height (in inches) of each facet.
    aspect : scalar
        Aspect * height gives the width (in inches) of each facet.
    corner : bool
        If True, don't add axes to the upper (off-diagonal) triangle of the
        grid, making this a "corner" plot.
    dropna : boolean
        Drop missing values from the data before plotting.
    {plot, diag, grid}_kws : dicts
        Dictionaries of keyword arguments. ``plot_kws`` are passed to the
        bivariate plotting function, ``diag_kws`` are passed to the univariate
        plotting function, and ``grid_kws`` are passed to the :class:`PairGrid`
        constructor.

    Returns
    -------
    grid : :class:`PairGrid`
        Returns the underlying :class:`PairGrid` instance for further tweaking.

    See Also
    --------
    PairGrid : Subplot grid for more flexible plotting of pairwise relationships.
    JointGrid : Grid for plotting joint and marginal distributions of two variables.

    Examples
    --------

    .. include:: ../docstrings/pairplot.rst

    """
    # Avoid circular import
    from .distributions import histplot, kdeplot

    # Handle deprecations
    if size is not None:
        height = size
        msg = ("The `size` parameter has been renamed to `height`; "
               "please update your code.")
        warnings.warn(msg, UserWarning)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "'data' must be pandas DataFrame object, not: {typefound}".format(
                typefound=type(data)))

    plot_kws = {} if plot_kws is None else plot_kws.copy()
    diag_kws = {} if diag_kws is None else diag_kws.copy()
    grid_kws = {} if grid_kws is None else grid_kws.copy()

    # Resolve "auto" diag kind
    if diag_kind == "auto":
        if hue is None:
            diag_kind = "kde" if kind == "kde" else "hist"
        else:
            diag_kind = "hist" if kind == "hist" else "kde"

    # Set up the PairGrid
    grid_kws.setdefault("diag_sharey", diag_kind == "hist")
    grid = PairGrid(data, vars=vars, x_vars=x_vars, y_vars=y_vars, hue=hue,
                    hue_order=hue_order, palette=palette, corner=corner,
                    height=height, aspect=aspect, dropna=dropna, **grid_kws)

    # Add the markers here as PairGrid has figured out how many levels of the
    # hue variable are needed and we don't want to duplicate that process
    if markers is not None:
        if kind == "reg":
            # Needed until regplot supports style
            if grid.hue_names is None:
                n_markers = 1
            else:
                n_markers = len(grid.hue_names)
            if not isinstance(markers, list):
                markers = [markers] * n_markers
            if len(markers) != n_markers:
                raise ValueError(("markers must be a singleton or a list of "
                                  "markers for each level of the hue variable"))
            grid.hue_kws = {"marker": markers}
        elif kind == "scatter":
            if isinstance(markers, str):
                plot_kws["marker"] = markers
            elif hue is not None:
                plot_kws["style"] = data[hue]
                plot_kws["markers"] = markers

    # Draw the marginal plots on the diagonal
    diag_kws = diag_kws.copy()
    diag_kws.setdefault("legend", False)
    if diag_kind == "hist":
        grid.map_diag(histplot, **diag_kws)
    elif diag_kind == "kde":
        diag_kws.setdefault("fill", True)
        grid.map_diag(kdeplot, **diag_kws)

    # Maybe plot on the off-diagonals
    if diag_kind is not None:
        plotter = grid.map_offdiag
    else:
        plotter = grid.map

    if kind == "scatter":
        from .relational import scatterplot  # Avoid circular import
        plotter(scatterplot, **plot_kws)
    elif kind == "reg":
        from .regression import regplot  # Avoid circular import
        plotter(regplot, **plot_kws)
    elif kind == "kde":
        from .distributions import kdeplot  # Avoid circular import
        plotter(kdeplot, **plot_kws)
    elif kind == "hist":
        from .distributions import histplot  # Avoid circular import
        plotter(histplot, **plot_kws)

    # Add a legend
    if hue is not None:
        grid.add_legend()

    grid.tight_layout()

    return grid