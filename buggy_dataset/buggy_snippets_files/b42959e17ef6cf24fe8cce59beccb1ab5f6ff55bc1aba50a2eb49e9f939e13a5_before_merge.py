    def __setitem__(self, key, value):
        value = extract_array(value, extract_numpy=True)

        if not lib.is_scalar(key) and is_list_like(key):
            key = np.asarray(key)

        if not lib.is_scalar(value):
            value = np.asarray(value)

        values = self._ndarray
        t = np.result_type(value, values)
        if t != self._ndarray.dtype:
            values = values.astype(t, casting="safe")
            values[key] = value
            self._dtype = PandasDtype(t)
            self._ndarray = values
        else:
            self._ndarray[key] = value