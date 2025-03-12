def main():
    module = AnsibleModule(
        argument_spec=dict(
            api_key=dict(required=True, no_log=True),
            api_host=dict(required=False),
            app_key=dict(required=True, no_log=True),
            state=dict(required=True, choices=['present', 'absent', 'mute', 'unmute']),
            type=dict(required=False, choices=['metric alert', 'service check', 'event alert', 'process alert']),
            name=dict(required=True),
            query=dict(required=False),
            message=dict(required=False, default=None),
            silenced=dict(required=False, default=None, type='dict'),
            notify_no_data=dict(required=False, default=False, type='bool'),
            no_data_timeframe=dict(required=False, default=None),
            timeout_h=dict(required=False, default=None),
            renotify_interval=dict(required=False, default=None),
            escalation_message=dict(required=False, default=None),
            notify_audit=dict(required=False, default=False, type='bool'),
            thresholds=dict(required=False, type='dict', default=None),
            tags=dict(required=False, type='list', default=None),
            locked=dict(required=False, default=False, type='bool'),
            require_full_window=dict(required=False, default=None, type='bool'),
            new_host_delay=dict(required=False, default=None),
            evaluation_delay=dict(required=False, default=None),
            id=dict(required=False)
        )
    )

    # Prepare Datadog
    if not HAS_DATADOG:
        module.fail_json(msg=missing_required_lib('datadogpy'), exception=DATADOG_IMP_ERR)

    options = {
        'api_key': module.params['api_key'],
        'api_host': module.params['api_host'],
        'app_key': module.params['app_key']
    }

    initialize(**options)

    # Check if api_key and app_key is correct or not
    # if not, then fail here.
    response = api.Monitor.get_all()
    if isinstance(response, dict):
        msg = response.get('errors', None)
        if msg:
            module.fail_json(msg="Failed to connect Datadog server using given app_key and api_key : {0}".format(msg[0]))

    if module.params['state'] == 'present':
        install_monitor(module)
    elif module.params['state'] == 'absent':
        delete_monitor(module)
    elif module.params['state'] == 'mute':
        mute_monitor(module)
    elif module.params['state'] == 'unmute':
        unmute_monitor(module)