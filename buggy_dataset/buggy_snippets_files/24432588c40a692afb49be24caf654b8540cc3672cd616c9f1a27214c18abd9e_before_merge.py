def get_switchport(name, module):
    config = run_commands(module, ['show interface {0} switchport'.format(name)])[0]
    mode = re.search(r'Administrative Mode: (?:.* )?(\w+)$', config, re.M).group(1)
    access = re.search(r'Access Mode VLAN: (\d+)', config).group(1)
    native = re.search(r'Trunking Native Mode VLAN: (\d+)', config).group(1)
    trunk = re.search(r'Trunking VLANs Enabled: (.+)$', config, re.M).group(1)
    if trunk == 'ALL':
        trunk = '1-4094'

    switchport_config = {
        "interface": name,
        "mode": mode,
        "access_vlan": access,
        "native_vlan": native,
        "trunk_vlans": trunk,
    }

    return switchport_config