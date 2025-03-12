def load_config(module, command_filter, commit=False, replace=False,
                comment=None, admin=False, running=None, nc_get_filter=None):

    conn = get_connection(module)

    diff = None
    if is_netconf(module):
        # FIXME: check for platform behaviour and restore this
        # conn.lock(target = 'candidate')
        # conn.discard_changes()

        try:
            for filter in to_list(command_filter):
                conn.edit_config(filter)

            candidate = get_config(module, source='candidate', config_filter=nc_get_filter)
            diff = get_config_diff(module, running, candidate)

            if commit and diff:
                commit_config(module)
            else:
                discard_config(module)
        finally:
            # conn.unlock(target = 'candidate')
            pass

    elif is_cliconf(module):
        # to keep the pre-cliconf behaviour, make a copy, avoid adding commands to input list
        cmd_filter = deepcopy(command_filter)
        cmd_filter.insert(0, 'configure terminal')
        if admin:
            cmd_filter.insert(0, 'admin')
        conn.edit_config(cmd_filter)

        if module._diff:
            diff = get_config_diff(module)

        if replace:
            cmd = list()
            cmd.append({'command': 'commit replace',
                        'prompt': 'This commit will replace or remove the entire running configuration',
                        'answer': 'yes'})
            cmd.append('end')
            conn.edit_config(cmd)
        elif commit:
            commit_config(module, comment=comment)
            conn.edit_config('end')
            if admin:
                conn.edit_config('exit')
        else:
            conn.discard_changes()

    return diff