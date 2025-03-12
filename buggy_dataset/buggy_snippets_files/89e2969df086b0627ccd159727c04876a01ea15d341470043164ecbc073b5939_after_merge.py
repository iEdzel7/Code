def pivot_table(data, values=None, index=None, columns=None, aggfunc='mean',
                fill_value=None, margins=False, dropna=True,
                margins_name='All'):
    index = _convert_by(index)
    columns = _convert_by(columns)

    if isinstance(aggfunc, list):
        pieces = []
        keys = []
        for func in aggfunc:
            table = pivot_table(data, values=values, index=index,
                                columns=columns,
                                fill_value=fill_value, aggfunc=func,
                                margins=margins, margins_name=margins_name)
            pieces.append(table)
            keys.append(getattr(func, '__name__', func))

        return concat(pieces, keys=keys, axis=1)

    keys = index + columns

    values_passed = values is not None
    if values_passed:
        if is_list_like(values):
            values_multi = True
            values = list(values)
        else:
            values_multi = False
            values = [values]

        # GH14938 Make sure value labels are in data
        for i in values:
            if i not in data:
                raise KeyError(i)

        to_filter = []
        for x in keys + values:
            if isinstance(x, Grouper):
                x = x.key
            try:
                if x in data:
                    to_filter.append(x)
            except TypeError:
                pass
        if len(to_filter) < len(data.columns):
            data = data[to_filter]

    else:
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)

    grouped = data.groupby(keys)
    agged = grouped.agg(aggfunc)

    table = agged
    if table.index.nlevels > 1:
        # Related GH #17123
        # If index_names are integers, determine whether the integers refer
        # to the level position or name.
        index_names = agged.index.names[:len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if name is None or name in index_names:
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        table = agged.unstack(to_unstack)

    if not dropna:
        from pandas import MultiIndex
        try:
            m = MultiIndex.from_arrays(cartesian_product(table.index.levels),
                                       names=table.index.names)
            table = table.reindex(m, axis=0)
        except AttributeError:
            pass  # it's a single level

        try:
            m = MultiIndex.from_arrays(cartesian_product(table.columns.levels),
                                       names=table.columns.names)
            table = table.reindex(m, axis=1)
        except AttributeError:
            pass  # it's a single level or a series

    if isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)

    if fill_value is not None:
        table = table.fillna(value=fill_value, downcast='infer')

    if margins:
        if dropna:
            data = data[data.notna().all(axis=1)]
        table = _add_margins(table, data, values, rows=index,
                             cols=columns, aggfunc=aggfunc,
                             margins_name=margins_name, fill_value=fill_value)

    # discard the top level
    if values_passed and not values_multi and not table.empty and \
       (table.columns.nlevels > 1):
        table = table[values[0]]

    if len(index) == 0 and len(columns) > 0:
        table = table.T

    # GH 15193 Make sure empty columns are removed if dropna=True
    if isinstance(table, ABCDataFrame) and dropna:
        table = table.dropna(how='all', axis=1)

    return table