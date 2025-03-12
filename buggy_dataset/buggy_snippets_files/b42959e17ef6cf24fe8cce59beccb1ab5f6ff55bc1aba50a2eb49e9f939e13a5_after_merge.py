    def __setitem__(self, key, value):
        value = extract_array(value, extract_numpy=True)

        if not lib.is_scalar(key) and is_list_like(key):
            key = np.asarray(key)

        if not lib.is_scalar(value):
            value = np.asarray(value)

        value = np.asarray(value, dtype=self._ndarray.dtype)
        self._ndarray[key] = value