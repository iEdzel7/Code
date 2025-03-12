def map_obj_to_ele(module, want):
    element = Element('system')
    login = SubElement(element, 'login')

    for item in want:
        if item['state'] != 'present':
            if item['name'] == 'root':
                module.fail_json(msg="cannot delete the 'root' account.")
            operation = 'delete'
        else:
            operation = 'merge'

        user = SubElement(login, 'user', {'operation': operation})

        SubElement(user, 'name').text = item['name']

        if operation == 'merge':
            if item['active']:
                user.set('active', 'active')
            else:
                user.set('inactive', 'inactive')

            if item['role']:
                SubElement(user, 'class').text = item['role']

            if item.get('full_name'):
                SubElement(user, 'full-name').text = item['full_name']

            if item.get('sshkey'):
                auth = SubElement(user, 'authentication')
                if 'ssh-rsa' in item['sshkey']:
                    ssh_rsa = SubElement(auth, 'ssh-rsa')
                elif 'ssh-dss' in item['sshkey']:
                    ssh_rsa = SubElement(auth, 'ssh-dsa')
                elif 'ecdsa-sha2' in item['sshkey']:
                    ssh_rsa = SubElement(auth, 'ssh-ecdsa')
                elif 'ssh-ed25519' in item['sshkey']:
                    ssh_rsa = SubElement(auth, 'ssh-ed25519')
                key = SubElement(ssh_rsa, 'name').text = item['sshkey']

            if item.get('encrypted_password'):
                if 'auth' not in locals():
                    auth = SubElement(user, 'authentication')
                SubElement(auth, 'encrypted-password').text = item['encrypted_password']

    return element