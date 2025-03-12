    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array or IntegerArray with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray or IntegerArray
            NumPy ndarray or IntergerArray with 'dtype' for its dtype.

        Raises
        ------
        TypeError
            if incompatible type with an IntegerDtype, equivalent of same_kind
            casting
        """
        from pandas.core.arrays.boolean import BooleanArray, BooleanDtype

        dtype = pandas_dtype(dtype)

        # if we are astyping to an existing IntegerDtype we can fastpath
        if isinstance(dtype, _IntegerDtype):
            result = self._data.astype(dtype.numpy_dtype, copy=False)
            return type(self)(result, mask=self._mask, copy=False)
        elif isinstance(dtype, BooleanDtype):
            result = self._data.astype("bool", copy=False)
            return BooleanArray(result, mask=self._mask, copy=False)

        # coerce
        if is_float_dtype(dtype):
            # In astype, we consider dtype=float to also mean na_value=np.nan
            kwargs = dict(na_value=np.nan)
        else:
            kwargs = {}

        data = self.to_numpy(dtype=dtype, **kwargs)
        return astype_nansafe(data, dtype, copy=False)