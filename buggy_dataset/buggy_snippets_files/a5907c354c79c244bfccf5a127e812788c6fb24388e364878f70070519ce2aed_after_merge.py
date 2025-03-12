def remove_values(value, no_log_strings):
    """ Remove strings in no_log_strings from value.  If value is a container
    type, then remove a lot more"""
    deferred_removals = deque()

    no_log_strings = [to_native(s, errors='surrogate_or_strict') for s in no_log_strings]
    new_value = _remove_values_conditions(value, no_log_strings, deferred_removals)

    while deferred_removals:
        old_data, new_data = deferred_removals.popleft()
        if isinstance(new_data, Mapping):
            for old_key, old_elem in old_data.items():
                new_elem = _remove_values_conditions(old_elem, no_log_strings, deferred_removals)
                new_data[old_key] = new_elem
        else:
            for elem in old_data:
                new_elem = _remove_values_conditions(elem, no_log_strings, deferred_removals)
                if isinstance(new_data, MutableSequence):
                    new_data.append(new_elem)
                elif isinstance(new_data, MutableSet):
                    new_data.add(new_elem)
                else:
                    raise TypeError('Unknown container type encountered when removing private values from output')

    return new_value