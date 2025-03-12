    def pandas(tclass, *targs, **tkwargs):
        """
        Registers the given `tqdm` class with
            pandas.core.
            ( frame.DataFrame
            | series.Series
            | groupby.DataFrameGroupBy
            | groupby.SeriesGroupBy
            ).progress_apply

        A new instance will be create every time `progress_apply` is called,
        and each instance will automatically close() upon completion.

        Parameters
        ----------
        targs, tkwargs  : arguments for the tqdm instance

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from tqdm import tqdm, tqdm_gui
        >>>
        >>> df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
        >>> tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc
        >>> # Now you can use `progress_apply` instead of `apply`
        >>> df.groupby(0).progress_apply(lambda x: x**2)

        References
        ----------
        https://stackoverflow.com/questions/18603270/
        progress-indicator-during-pandas-operations-python
        """
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
        from pandas.core.groupby import DataFrameGroupBy
        from pandas.core.groupby import SeriesGroupBy
        from pandas.core.groupby import GroupBy
        from pandas.core.groupby import PanelGroupBy
        from pandas import Panel

        deprecated_t = [tkwargs.pop('deprecated_t', None)]

        def inner_generator(df_function='apply'):
            def inner(df, func, *args, **kwargs):
                """
                Parameters
                ----------
                df  : (DataFrame|Series)[GroupBy]
                    Data (may be grouped).
                func  : function
                    To be applied on the (grouped) data.
                *args, *kwargs  : optional
                    Transmitted to `df.apply()`.
                """
                # Precompute total iterations
                total = getattr(df, 'ngroups', None)
                if total is None:  # not grouped
                    total = len(df) if isinstance(df, Series) \
                        else df.size // len(df)
                else:
                    total += 1  # pandas calls update once too many

                # Init bar
                if deprecated_t[0] is not None:
                    t = deprecated_t[0]
                    deprecated_t[0] = None
                else:
                    t = tclass(*targs, total=total, **tkwargs)

                # Define bar updating wrapper
                def wrapper(*args, **kwargs):
                    t.update()
                    return func(*args, **kwargs)

                # Apply the provided function (in *args and **kwargs)
                # on the df using our wrapper (which provides bar updating)
                result = getattr(df, df_function)(wrapper, *args, **kwargs)

                # Close bar and return pandas calculation result
                t.close()
                return result
            return inner

        # Monkeypatch pandas to provide easy methods
        # Enable custom tqdm progress in pandas!
        Series.progress_apply = inner_generator()
        SeriesGroupBy.progress_apply = inner_generator()
        Series.progress_map = inner_generator('map')
        SeriesGroupBy.progress_map = inner_generator('map')

        DataFrame.progress_apply = inner_generator()
        DataFrameGroupBy.progress_apply = inner_generator()
        DataFrame.progress_applymap = inner_generator('applymap')

        Panel.progress_apply = inner_generator()
        PanelGroupBy.progress_apply = inner_generator()

        GroupBy.progress_apply = inner_generator()
        GroupBy.progress_aggregate = inner_generator('aggregate')
        GroupBy.progress_transform = inner_generator('transform')