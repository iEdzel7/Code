    def __new__(
        cls,
        data=None,
        unit=None,
        freq=None,
        closed=None,
        dtype=_TD_DTYPE,
        copy=False,
        name=None,
    ):

        if is_scalar(data):
            raise TypeError(
                f"{cls.__name__}() must be called with a "
                f"collection of some kind, {repr(data)} was passed"
            )

        if unit in {"Y", "y", "M"}:
            raise ValueError(
                "Units 'M' and 'Y' are no longer supported, as they do not "
                "represent unambiguous timedelta values durations."
            )

        if isinstance(data, TimedeltaArray):
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name, freq=freq)

        if isinstance(data, TimedeltaIndex) and freq is None and name is None:
            if copy:
                return data.copy()
            else:
                return data._shallow_copy()

        # - Cases checked above all return/raise before reaching here - #

        tdarr = TimedeltaArray._from_sequence(
            data, freq=freq, unit=unit, dtype=dtype, copy=copy
        )
        return cls._simple_new(tdarr._data, freq=tdarr.freq, name=name)