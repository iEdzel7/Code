def main():
    argument_spec = dict(
        name=dict(required=True),
        description=dict(),
        organization=dict(required=True),
        variables=dict(),
        kind=dict(choices=['', 'smart'], default=''),
        host_filter=dict(),
        state=dict(choices=['present', 'absent'], default='present'),
    )

    module = TowerModule(argument_spec=argument_spec, supports_check_mode=True)

    name = module.params.get('name')
    description = module.params.get('description')
    organization = module.params.get('organization')
    variables = module.params.get('variables')
    state = module.params.get('state')
    kind = module.params.get('kind')
    host_filter = module.params.get('host_filter')

    json_output = {'inventory': name, 'state': state}

    tower_auth = tower_auth_config(module)
    with settings.runtime_values(**tower_auth):
        tower_check_mode(module)
        inventory = tower_cli.get_resource('inventory')

        try:
            org_res = tower_cli.get_resource('organization')
            org = org_res.get(name=organization)

            if state == 'present':
                result = inventory.modify(name=name, organization=org['id'], variables=variables,
                                          description=description, kind=kind, host_filter=host_filter,
                                          create_on_missing=True)
                json_output['id'] = result['id']
            elif state == 'absent':
                result = inventory.delete(name=name, organization=org['id'])
        except (exc.NotFound) as excinfo:
            module.fail_json(msg='Failed to update inventory, organization not found: {0}'.format(excinfo), changed=False)
        except (exc.ConnectionError, exc.BadRequest, exc.AuthError) as excinfo:
            module.fail_json(msg='Failed to update inventory: {0}'.format(excinfo), changed=False)

    json_output['changed'] = result['changed']
    module.exit_json(**json_output)