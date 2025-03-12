def main():
    argument_spec = ovirt_full_argument_spec(
        state=dict(
            choices=[
                'present', 'absent', 'maintenance', 'upgraded', 'started',
                'restarted', 'stopped', 'reinstalled', 'iscsidiscover', 'iscsilogin'
            ],
            default='present',
        ),
        name=dict(required=True),
        comment=dict(default=None),
        cluster=dict(default=None),
        address=dict(default=None),
        password=dict(default=None, no_log=True),
        public_key=dict(default=False, type='bool', aliases=['ssh_public_key']),
        kdump_integration=dict(default=None, choices=['enabled', 'disabled']),
        spm_priority=dict(default=None, type='int'),
        override_iptables=dict(default=None, type='bool'),
        force=dict(default=False, type='bool'),
        timeout=dict(default=600, type='int'),
        override_display=dict(default=None),
        kernel_params=dict(default=None, type='list'),
        hosted_engine=dict(default=None, choices=['deploy', 'undeploy']),
        power_management_enabled=dict(default=None, type='bool'),
        activate=dict(default=True, type='bool'),
        iscsi=dict(default=None, type='dict'),
        check_upgrade=dict(default=True, type='bool'),
        reboot_after_upgrade=dict(default=True, type='bool'),
    )
    module = AnsibleModule(
        argument_spec=argument_spec,
        supports_check_mode=True,
        required_if=[
            ['state', 'iscsidiscover', ['iscsi']],
            ['state', 'iscsilogin', ['iscsi']]
        ]
    )
    check_sdk(module)

    try:
        auth = module.params.pop('auth')
        connection = create_connection(auth)
        hosts_service = connection.system_service().hosts_service()
        hosts_module = HostsModule(
            connection=connection,
            module=module,
            service=hosts_service,
        )

        state = module.params['state']
        host = control_state(hosts_module)
        if state == 'present':
            ret = hosts_module.create(
                deploy_hosted_engine=(
                    module.params.get('hosted_engine') == 'deploy'
                ) if module.params.get('hosted_engine') is not None else None,
                result_state=hoststate.UP if host is None else None,
                fail_condition=hosts_module.failed_state_after_reinstall if host is None else lambda h: False,
            )
            if module.params['activate'] and host is not None:
                ret = hosts_module.action(
                    action='activate',
                    action_condition=lambda h: h.status != hoststate.UP,
                    wait_condition=lambda h: h.status == hoststate.UP,
                    fail_condition=failed_state,
                )
        elif state == 'absent':
            ret = hosts_module.remove()
        elif state == 'maintenance':
            hosts_module.action(
                action='deactivate',
                action_condition=lambda h: h.status != hoststate.MAINTENANCE,
                wait_condition=lambda h: h.status == hoststate.MAINTENANCE,
                fail_condition=failed_state,
            )
            ret = hosts_module.create()
        elif state == 'upgraded':
            result_state = hoststate.MAINTENANCE if host.status == hoststate.MAINTENANCE else hoststate.UP
            events_service = connection.system_service().events_service()
            last_event = events_service.list(max=1)[0]

            if module.params['check_upgrade']:
                hosts_module.action(
                    action='upgrade_check',
                    action_condition=lambda host: not host.update_available,
                    wait_condition=lambda host: host.update_available or (
                        len([
                            event
                            for event in events_service.list(
                                from_=int(last_event.id),
                                search='type=885',
                                # Uncomment when 4.1 is EOL, and remove the cond:
                                # if host.name in event.description
                                # search='type=885 and host.name=%s' % host.name,
                            ) if host.name in event.description
                        ]) > 0
                    ),
                    fail_condition=lambda host: len([
                        event
                        for event in events_service.list(
                            from_=int(last_event.id),
                            search='type=839 or type=887 and host.name=%s' % host.name,
                        )
                    ]) > 0,
                )
                # Set to False, because upgrade_check isn't 'changing' action:
                hosts_module._changed = False
            ret = hosts_module.action(
                action='upgrade',
                action_condition=lambda h: h.update_available,
                wait_condition=lambda h: h.status == result_state,
                post_action=lambda h: time.sleep(module.params['poll_interval']),
                fail_condition=hosts_module.failed_state_after_reinstall,
                reboot=module.params['reboot_after_upgrade'],
            )
        elif state == 'iscsidiscover':
            host_id = get_id_by_name(hosts_service, module.params['name'])
            iscsi_param = module.params['iscsi']
            iscsi_targets = hosts_service.service(host_id).iscsi_discover(
                iscsi=otypes.IscsiDetails(
                    port=int(iscsi_param.get('port', 3260)),
                    username=iscsi_param.get('username'),
                    password=iscsi_param.get('password'),
                    address=iscsi_param.get('address'),
                ),
            )
            ret = {
                'changed': False,
                'id': host_id,
                'iscsi_targets': iscsi_targets,
            }
        elif state == 'iscsilogin':
            host_id = get_id_by_name(hosts_service, module.params['name'])
            iscsi_param = module.params['iscsi']
            ret = hosts_module.action(
                action='iscsi_login',
                iscsi=otypes.IscsiDetails(
                    port=int(iscsi_param.get('port', 3260)),
                    username=iscsi_param.get('username'),
                    password=iscsi_param.get('password'),
                    address=iscsi_param.get('address'),
                    target=iscsi_param.get('target'),
                ),
            )
        elif state == 'started':
            ret = hosts_module.action(
                action='fence',
                action_condition=lambda h: h.status == hoststate.DOWN,
                wait_condition=lambda h: h.status in [hoststate.UP, hoststate.MAINTENANCE],
                fail_condition=hosts_module.failed_state_after_reinstall,
                fence_type='start',
            )
        elif state == 'stopped':
            hosts_module.action(
                action='deactivate',
                action_condition=lambda h: h.status not in [hoststate.MAINTENANCE, hoststate.DOWN],
                wait_condition=lambda h: h.status in [hoststate.MAINTENANCE, hoststate.DOWN],
                fail_condition=failed_state,
            )
            ret = hosts_module.action(
                action='fence',
                action_condition=lambda h: h.status != hoststate.DOWN,
                wait_condition=lambda h: h.status == hoststate.DOWN if module.params['wait'] else True,
                fail_condition=failed_state,
                fence_type='stop',
            )
        elif state == 'restarted':
            ret = hosts_module.action(
                action='fence',
                wait_condition=lambda h: h.status == hoststate.UP,
                fail_condition=hosts_module.failed_state_after_reinstall,
                fence_type='restart',
            )
        elif state == 'reinstalled':
            # Deactivate host if not in maintanence:
            hosts_module.action(
                action='deactivate',
                action_condition=lambda h: h.status not in [hoststate.MAINTENANCE, hoststate.DOWN],
                wait_condition=lambda h: h.status in [hoststate.MAINTENANCE, hoststate.DOWN],
                fail_condition=failed_state,
            )

            # Reinstall host:
            hosts_module.action(
                action='install',
                action_condition=lambda h: h.status == hoststate.MAINTENANCE,
                post_action=hosts_module.post_reinstall,
                wait_condition=lambda h: h.status == hoststate.MAINTENANCE,
                fail_condition=hosts_module.failed_state_after_reinstall,
                host=otypes.Host(
                    override_iptables=module.params['override_iptables'],
                ) if module.params['override_iptables'] else None,
                root_password=module.params['password'],
                ssh=otypes.Ssh(
                    authentication_method=otypes.SshAuthenticationMethod.PUBLICKEY,
                ) if module.params['public_key'] else None,
                deploy_hosted_engine=(
                    module.params.get('hosted_engine') == 'deploy'
                ) if module.params.get('hosted_engine') is not None else None,
                undeploy_hosted_engine=(
                    module.params.get('hosted_engine') == 'undeploy'
                ) if module.params.get('hosted_engine') is not None else None,
            )

            # Activate host after reinstall:
            ret = hosts_module.action(
                action='activate',
                action_condition=lambda h: h.status == hoststate.MAINTENANCE,
                wait_condition=lambda h: h.status == hoststate.UP,
                fail_condition=failed_state,
            )
        module.exit_json(**ret)
    except Exception as e:
        module.fail_json(msg=str(e), exception=traceback.format_exc())
    finally:
        connection.close(logout=auth.get('token') is None)