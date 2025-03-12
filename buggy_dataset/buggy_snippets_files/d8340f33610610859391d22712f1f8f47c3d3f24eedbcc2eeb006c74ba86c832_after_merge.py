def main():
    """ main entry point for module execution
    """
    backup_spec = dict(
        filename=dict(),
        dir_path=dict(type='path')
    )
    argument_spec = dict(
        content=dict(aliases=['xml']),
        target=dict(choices=['auto', 'candidate', 'running'], default='auto', aliases=['datastore']),
        source_datastore=dict(aliases=['source']),
        format=dict(choices=['xml', 'text'], default='xml'),
        lock=dict(choices=['never', 'always', 'if-supported'], default='always'),
        default_operation=dict(choices=['merge', 'replace', 'none']),
        confirm=dict(type='int', default=0),
        confirm_commit=dict(type='bool', default=False),
        error_option=dict(choices=['stop-on-error', 'continue-on-error', 'rollback-on-error'], default='stop-on-error'),
        backup=dict(type='bool', default=False),
        backup_options=dict(type='dict', options=backup_spec),
        save=dict(type='bool', default=False),
        delete=dict(type='bool', default=False),
        commit=dict(type='bool', default=True),
        validate=dict(type='bool', default=False),
    )

    # deprecated options
    netconf_top_spec = {
        'src': dict(type='path', removed_in_version=2.11),
        'host': dict(removed_in_version=2.11),
        'port': dict(removed_in_version=2.11, type='int', default=830),
        'username': dict(fallback=(env_fallback, ['ANSIBLE_NET_USERNAME']), removed_in_version=2.11, no_log=True),
        'password': dict(fallback=(env_fallback, ['ANSIBLE_NET_PASSWORD']), removed_in_version=2.11, no_log=True),
        'ssh_keyfile': dict(fallback=(env_fallback, ['ANSIBLE_NET_SSH_KEYFILE']), removed_in_version=2.11, type='path'),
        'hostkey_verify': dict(removed_in_version=2.11, type='bool', default=True),
        'look_for_keys': dict(removed_in_version=2.11, type='bool', default=True),
        'timeout': dict(removed_in_version=2.11, type='int', default=10),
    }
    argument_spec.update(netconf_top_spec)

    mutually_exclusive = [('content', 'src', 'source', 'delete', 'confirm_commit')]
    required_one_of = [('content', 'src', 'source', 'delete', 'confirm_commit')]

    module = AnsibleModule(argument_spec=argument_spec,
                           required_one_of=required_one_of,
                           mutually_exclusive=mutually_exclusive,
                           supports_check_mode=True)

    if module.params['src']:
        module.deprecate(msg="argument 'src' has been deprecated. Use file lookup plugin instead to read file contents.",
                         version="2.11")

    config = module.params['content'] or module.params['src']
    target = module.params['target']
    lock = module.params['lock']
    source = module.params['source_datastore']
    delete = module.params['delete']
    confirm_commit = module.params['confirm_commit']
    confirm = module.params['confirm']
    validate = module.params['validate']
    save = module.params['save']

    conn = Connection(module._socket_path)
    capabilities = get_capabilities(module)
    operations = capabilities['device_operations']

    supports_commit = operations.get('supports_commit', False)
    supports_writable_running = operations.get('supports_writable_running', False)
    supports_startup = operations.get('supports_startup', False)

    # identify target datastore
    if target == 'candidate' and not supports_commit:
        module.fail_json(msg=':candidate is not supported by this netconf server')
    elif target == 'running' and not supports_writable_running:
        module.fail_json(msg=':writable-running is not supported by this netconf server')
    elif target == 'auto':
        if supports_commit:
            target = 'candidate'
        elif supports_writable_running:
            target = 'running'
        else:
            module.fail_json(msg='neither :candidate nor :writable-running are supported by this netconf server')

    # Netconf server capability validation against input options
    if save and not supports_startup:
        module.fail_json(msg='cannot copy <%s/> to <startup/>, while :startup is not supported' % target)

    if confirm_commit and not operations.get('supports_confirm_commit', False):
        module.fail_json(msg='confirm commit is not supported by Netconf server')

    if (confirm > 0) and not operations.get('supports_confirm_commit', False):
        module.fail_json(msg='confirm commit is not supported by this netconf server, given confirm timeout: %d' % confirm)

    if validate and not operations.get('supports_validate', False):
        module.fail_json(msg='validate is not supported by this netconf server')

    if lock == 'never':
        execute_lock = False
    elif target in operations.get('lock_datastore', []):
        # lock is requested (always/if-support) and supported => lets do it
        execute_lock = True
    else:
        # lock is requested (always/if-supported) but not supported => issue warning
        module.warn("lock operation on '%s' source is not supported on this device" % target)
        execute_lock = (lock == 'always')

    result = {'changed': False, 'server_capabilities': capabilities.get('server_capabilities', [])}
    before = None
    after = None
    locked = False
    try:
        if module.params['backup']:
            response = get_config(module, target, lock=execute_lock)
            before = to_text(response, errors='surrogate_then_replace').strip()
            result['__backup__'] = before.strip()
        if validate:
            conn.validate(target)
        if source:
            if not module.check_mode:
                conn.copy(source, target)
            result['changed'] = True
        elif delete:
            if not module.check_mode:
                conn.delete(target)
            result['changed'] = True
        elif confirm_commit:
            if not module.check_mode:
                conn.commit()
            result['changed'] = True
        elif config:
            if module.check_mode and not supports_commit:
                module.warn("check mode not supported as Netconf server doesn't support candidate capability")
                result['changed'] = True
                module.exit_json(**result)

            if execute_lock:
                conn.lock(target=target)
                locked = True
            if before is None:
                before = to_text(conn.get_config(source=target), errors='surrogate_then_replace').strip()

            kwargs = {
                'config': config,
                'target': target,
                'default_operation': module.params['default_operation'],
                'error_option': module.params['error_option'],
                'format': module.params['format'],
            }

            conn.edit_config(**kwargs)

            if supports_commit and module.params['commit']:
                after = to_text(conn.get_config(source='candidate'), errors='surrogate_then_replace').strip()
                if not module.check_mode:
                    confirm_timeout = confirm if confirm > 0 else None
                    confirmed_commit = True if confirm_timeout else False
                    conn.commit(confirmed=confirmed_commit, timeout=confirm_timeout)
                else:
                    conn.discard_changes()

            if after is None:
                after = to_text(conn.get_config(source='running'), errors='surrogate_then_replace').strip()

            sanitized_before = sanitize_xml(before)
            sanitized_after = sanitize_xml(after)
            if sanitized_before != sanitized_after:
                result['changed'] = True

            if result['changed']:
                if save and not module.check_mode:
                    conn.copy_config(target, 'startup')
                if module._diff:
                    result['diff'] = {'before': sanitized_before, 'after': sanitized_after}

    except ConnectionError as e:
        module.fail_json(msg=to_text(e, errors='surrogate_then_replace').strip())
    finally:
        if locked:
            conn.unlock(target=target)

    module.exit_json(**result)