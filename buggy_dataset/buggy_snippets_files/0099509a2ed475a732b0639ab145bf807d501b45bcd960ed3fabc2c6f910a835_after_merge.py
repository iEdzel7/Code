def get_defaults_flag(module):
    rc, out, err = exec_command(module, 'show running-config ?')
    out = to_text(out, errors='surrogate_then_replace')

    commands = set()
    for line in out.splitlines():
        if line.strip():
            commands.add(line.strip().split()[0])

    if 'all' in commands:
        return ['all']
    else:
        return ['full']