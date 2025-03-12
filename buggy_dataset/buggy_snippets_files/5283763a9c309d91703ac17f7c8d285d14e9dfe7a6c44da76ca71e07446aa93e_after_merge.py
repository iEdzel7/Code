def main():
    module = AnsibleModule(
        argument_spec = dict(
            state = dict(default='present', choices=['present', 'absent'], type='str'),
            name = dict(required=True, type='str'),
            login_user = dict(default='guest', type='str'),
            login_password = dict(default='guest', type='str', no_log=True),
            login_host = dict(default='localhost', type='str'),
            login_port = dict(default='15672', type='str'),
            vhost = dict(default='/', type='str'),
            durable = dict(default=True, type='bool'),
            auto_delete = dict(default=False, type='bool'),
            internal = dict(default=False, type='bool'),
            exchange_type = dict(default='direct', aliases=['type'], type='str'),
            arguments = dict(default=dict(), type='dict')
        ),
        supports_check_mode = True
    )

    url = "http://%s:%s/api/exchanges/%s/%s" % (
        module.params['login_host'],
        module.params['login_port'],
        urllib.quote(module.params['vhost'],''),
        urllib.quote(module.params['name'],'')
    )

    if not HAS_REQUESTS:
        module.fail_json(msg="requests library is required for this module. To install, use `pip install requests`")

    # Check if exchange already exists
    r = requests.get( url, auth=(module.params['login_user'],module.params['login_password']))

    if r.status_code==200:
        exchange_exists = True
        response = r.json()
    elif r.status_code==404:
        exchange_exists = False
        response = r.text
    else:
        module.fail_json(
            msg = "Invalid response from RESTAPI when trying to check if exchange exists",
            details = r.text
        )

    if module.params['state']=='present':
        change_required = not exchange_exists
    else:
        change_required = exchange_exists

    # Check if attributes change on existing exchange
    if not change_required and r.status_code==200 and module.params['state'] == 'present':
        if not (
            response['durable'] == module.params['durable'] and
            response['auto_delete'] == module.params['auto_delete'] and
            response['internal'] == module.params['internal'] and
            response['type'] == module.params['exchange_type']
        ):
            module.fail_json(
                msg = "RabbitMQ RESTAPI doesn't support attribute changes for existing exchanges"
            )

    # Exit if check_mode
    if module.check_mode:
        module.exit_json(
            changed= change_required,
            name = module.params['name'],
            details = response,
            arguments = module.params['arguments']
        )

    # Do changes
    if change_required:
        if module.params['state'] == 'present':
            r = requests.put(
                url,
                auth = (module.params['login_user'],module.params['login_password']),
                headers = { "content-type": "application/json"},
                data = json.dumps({
                    "durable": module.params['durable'],
                    "auto_delete": module.params['auto_delete'],
                    "internal": module.params['internal'],
                    "type": module.params['exchange_type'],
                    "arguments": module.params['arguments']
                    })
            )
        elif module.params['state'] == 'absent':
            r = requests.delete( url, auth = (module.params['login_user'],module.params['login_password']))

        # RabbitMQ 3.6.7 changed this response code from 204 to 201
        if r.status_code == 204 or r.status_code == 201:
            module.exit_json(
                changed = True,
                name = module.params['name']
            )
        else:
            module.fail_json(
                msg = "Error creating exchange",
                status = r.status_code,
                details = r.text
            )

    else:
        module.exit_json(
            changed = False,
            name = module.params['name']
        )