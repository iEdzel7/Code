    def pct_change(self, periods=1, fill_method="pad", limit=None, freq=None, **kwargs):
        # TODO: Not sure if above is correct - need someone to confirm.
        axis = self._get_axis_number(kwargs.pop("axis", self._stat_axis_name))
        if fill_method is None:
            data = self
        else:
            data = self.fillna(method=fill_method, limit=limit, axis=axis)

        rs = data.div(data.shift(periods=periods, freq=freq, axis=axis, **kwargs)) - 1
        rs = rs.loc[~rs.index.duplicated()]
        rs = rs.reindex_like(data)
        if freq is None:
            mask = isna(com.values_from_object(data))
            np.putmask(rs.values, mask, np.nan)
        return rs