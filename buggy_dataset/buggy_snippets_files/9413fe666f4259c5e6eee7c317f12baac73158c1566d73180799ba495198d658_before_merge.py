    def quantile(self, q=0.5, interpolation: str = "linear"):
        """
        Return group values at the given quantile, a la numpy.percentile.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value(s) between 0 and 1 providing the quantile(s) to compute.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Method to use when the desired quantile falls between two points.

        Returns
        -------
        Series or DataFrame
            Return type determined by caller of GroupBy object.

        See Also
        --------
        Series.quantile : Similar method for Series.
        DataFrame.quantile : Similar method for DataFrame.
        numpy.percentile : NumPy method to compute qth percentile.

        Examples
        --------
        >>> df = pd.DataFrame([
        ...     ['a', 1], ['a', 2], ['a', 3],
        ...     ['b', 1], ['b', 3], ['b', 5]
        ... ], columns=['key', 'val'])
        >>> df.groupby('key').quantile()
            val
        key
        a    2.0
        b    3.0
        """
        from pandas import concat

        def pre_processor(vals: np.ndarray) -> Tuple[np.ndarray, Optional[Type]]:
            if is_object_dtype(vals):
                raise TypeError(
                    "'quantile' cannot be performed against 'object' dtypes!"
                )

            inference = None
            if is_integer_dtype(vals):
                inference = np.int64
            elif is_datetime64_dtype(vals):
                inference = "datetime64[ns]"
                vals = np.asarray(vals).astype(np.float)

            return vals, inference

        def post_processor(vals: np.ndarray, inference: Optional[Type]) -> np.ndarray:
            if inference:
                # Check for edge case
                if not (
                    is_integer_dtype(inference)
                    and interpolation in {"linear", "midpoint"}
                ):
                    vals = vals.astype(inference)

            return vals

        if is_scalar(q):
            return self._get_cythonized_result(
                "group_quantile",
                aggregate=True,
                needs_values=True,
                needs_mask=True,
                cython_dtype=np.dtype(np.float64),
                pre_processing=pre_processor,
                post_processing=post_processor,
                q=q,
                interpolation=interpolation,
            )
        else:
            results = [
                self._get_cythonized_result(
                    "group_quantile",
                    aggregate=True,
                    needs_values=True,
                    needs_mask=True,
                    cython_dtype=np.dtype(np.float64),
                    pre_processing=pre_processor,
                    post_processing=post_processor,
                    q=qi,
                    interpolation=interpolation,
                )
                for qi in q
            ]
            result = concat(results, axis=0, keys=q)
            # fix levels to place quantiles on the inside
            # TODO(GH-10710): Ideally, we could write this as
            #  >>> result.stack(0).loc[pd.IndexSlice[:, ..., q], :]
            #  but this hits https://github.com/pandas-dev/pandas/issues/10710
            #  which doesn't reorder the list-like `q` on the inner level.
            order = list(range(1, result.index.nlevels)) + [0]

            # temporarily saves the index names
            index_names = np.array(result.index.names)

            # set index names to positions to avoid confusion
            result.index.names = np.arange(len(index_names))

            # place quantiles on the inside
            result = result.reorder_levels(order)

            # restore the index names in order
            result.index.names = index_names[order]

            # reorder rows to keep things sorted
            indices = np.arange(len(result)).reshape([len(q), self.ngroups]).T.flatten()
            return result.take(indices)