def map_config_to_obj(module):
    obj = []
    dest_group = ('console', 'host', 'monitor', 'buffered', 'on')

    data = get_config(module, flags=['| section logging'])

    for line in data.split('\n'):
        match = re.search(r'logging (\S+)', line, re.M)

        if match.group(1) in dest_group:
            dest = match.group(1)
        else:
            pass

        obj.append({'dest': dest,
                    'name': parse_name(line, dest),
                    'size': parse_size(line, dest),
                    'facility': parse_facility(line),
                    'level': parse_level(line, dest)})

    return obj