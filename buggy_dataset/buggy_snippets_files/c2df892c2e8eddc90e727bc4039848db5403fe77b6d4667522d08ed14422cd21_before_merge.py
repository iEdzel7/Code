def main():
    argument_spec = dict(
        name=dict(required=True),
        description=dict(required=False),
        extra_vars=dict(required=False),
        organization=dict(required=False),
        allow_simultaneous=dict(type='bool', required=False),
        schema=dict(required=False),
        survey=dict(required=False),
        survey_enabled=dict(type='bool', required=False),
        state=dict(choices=['present', 'absent'], default='present'),
    )

    module = TowerModule(
        argument_spec=argument_spec,
        supports_check_mode=False
    )

    name = module.params.get('name')
    state = module.params.get('state')

    schema = None
    if module.params.get('schema'):
        schema = module.params.get('schema')

    if schema and state == 'absent':
        module.fail_json(
            msg='Setting schema when state is absent is not allowed',
            changed=False
        )

    json_output = {'workflow_template': name, 'state': state}

    tower_auth = tower_auth_config(module)
    with settings.runtime_values(**tower_auth):
        tower_check_mode(module)
        wfjt_res = tower_cli.get_resource('workflow')
        params = {}
        params['name'] = name

        if module.params.get('description'):
            params['description'] = module.params.get('description')

        if module.params.get('organization'):
            organization_res = tower_cli.get_resource('organization')
            try:
                organization = organization_res.get(
                    name=module.params.get('organization'))
                params['organization'] = organization['id']
            except exc.NotFound as excinfo:
                module.fail_json(
                    msg='Failed to update organization source,'
                    'organization not found: {0}'.format(excinfo),
                    changed=False
                )

        if module.params.get('survey'):
            params['survey_spec'] = module.params.get('survey')

        for key in ('allow_simultaneous', 'extra_vars', 'survey_enabled',
                    'description'):
            if module.params.get(key):
                params[key] = module.params.get(key)

        try:
            if state == 'present':
                params['create_on_missing'] = True
                result = wfjt_res.modify(**params)
                json_output['id'] = result['id']
                if schema:
                    wfjt_res.schema(result['id'], schema)
            elif state == 'absent':
                params['fail_on_missing'] = False
                result = wfjt_res.delete(**params)
        except (exc.ConnectionError, exc.BadRequest) as excinfo:
            module.fail_json(msg='Failed to update workflow template: \
                    {0}'.format(excinfo), changed=False)

    json_output['changed'] = result['changed']
    module.exit_json(**json_output)