    def rename(self, index=None, **kwargs):
        """
        Alter Series index labels or name.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don't throw an
        error.

        Alternatively, change ``Series.name`` with a scalar value.

        See the :ref:`user guide <basics.rename>` for more.

        Parameters
        ----------
        index : scalar, hashable sequence, dict-like or function, optional
            dict-like or functions are transformations to apply to
            the index.
            Scalar or hashable sequence-like will alter the ``Series.name``
            attribute.
        copy : bool, default True
            Whether to copy underlying data.
        inplace : bool, default False
            Whether to return a new Series. If True then value of copy is
            ignored.
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified
            level.

        Returns
        -------
        Series
            Series with index labels or name altered.

        See Also
        --------
        Series.rename_axis : Set the name of the axis.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        dtype: int64
        >>> s.rename("my_name")  # scalar, changes Series.name
        0    1
        1    2
        2    3
        Name: my_name, dtype: int64
        >>> s.rename(lambda x: x ** 2)  # function, changes labels
        0    1
        1    2
        4    3
        dtype: int64
        >>> s.rename({1: 3, 2: 5})  # mapping, changes labels
        0    1
        3    2
        5    3
        dtype: int64
        """
        kwargs["inplace"] = validate_bool_kwarg(kwargs.get("inplace", False), "inplace")

        if callable(index) or is_dict_like(index):
            return super().rename(index=index, **kwargs)
        else:
            return self._set_name(index, inplace=kwargs.get("inplace"))