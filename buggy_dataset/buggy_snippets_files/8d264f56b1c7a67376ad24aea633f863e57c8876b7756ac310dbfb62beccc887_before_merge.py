def main():
    argument_spec = ec2_argument_spec()
    argument_spec.update(dict(
        name=dict(),
        group_id=dict(),
        description=dict(),
        vpc_id=dict(),
        rules=dict(type='list'),
        rules_egress=dict(type='list'),
        state=dict(default='present', type='str', choices=['present', 'absent']),
        purge_rules=dict(default=True, required=False, type='bool'),
        purge_rules_egress=dict(default=True, required=False, type='bool'),
        tags=dict(required=False, type='dict', aliases=['resource_tags']),
        purge_tags=dict(default=True, required=False, type='bool')
    )
    )
    module = AnsibleModule(
        argument_spec=argument_spec,
        supports_check_mode=True,
        required_one_of=[['name', 'group_id']],
        required_if=[['state', 'present', ['name']]],
    )

    if not HAS_BOTO3:
        module.fail_json(msg='boto3 required for this module')

    name = module.params['name']
    group_id = module.params['group_id']
    description = module.params['description']
    vpc_id = module.params['vpc_id']
    rules = deduplicate_rules_args(rules_expand_sources(rules_expand_ports(module.params['rules'])))
    rules_egress = deduplicate_rules_args(rules_expand_sources(rules_expand_ports(module.params['rules_egress'])))
    state = module.params.get('state')
    purge_rules = module.params['purge_rules']
    purge_rules_egress = module.params['purge_rules_egress']
    tags = module.params['tags']
    purge_tags = module.params['purge_tags']

    if state == 'present' and not description:
        module.fail_json(msg='Must provide description when state is present.')

    changed = False
    region, ec2_url, aws_connect_params = get_aws_connection_info(module, boto3=True)
    if not region:
        module.fail_json(msg="The AWS region must be specified as an "
                             "environment variable or in the AWS credentials "
                             "profile.")
    client = boto3_conn(module, conn_type='client', resource='ec2', endpoint=ec2_url, region=region, **aws_connect_params)
    group = None
    groups = dict()
    security_groups = []
    # do get all security groups
    # find if the group is present
    try:
        response = get_security_groups_with_backoff(client)
        security_groups = response.get('SecurityGroups', [])
    except botocore.exceptions.NoCredentialsError as e:
        module.fail_json(msg="Error in describe_security_groups: %s" % "Unable to locate credentials", exception=traceback.format_exc())
    except botocore.exceptions.ClientError as e:
        module.fail_json(msg="Error in describe_security_groups: %s" % e, exception=traceback.format_exc(),
                         **camel_dict_to_snake_dict(e.response))

    for sg in security_groups:
        groups[sg['GroupId']] = sg
        groupName = sg['GroupName']
        if groupName in groups:
            # Prioritise groups from the current VPC
            if vpc_id is None or sg['VpcId'] == vpc_id:
                groups[groupName] = sg
        else:
            groups[groupName] = sg

        if group_id:
            if sg['GroupId'] == group_id:
                group = sg
        else:
            if groupName == name and (vpc_id is None or sg['VpcId'] == vpc_id):
                group = sg

    # Ensure requested group is absent
    if state == 'absent':
        if group:
            # found a match, delete it
            try:
                if not module.check_mode:
                    client.delete_security_group(GroupId=group['GroupId'])
            except botocore.exceptions.ClientError as e:
                module.fail_json(msg="Unable to delete security group '%s' - %s" % (group, e),
                                 exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
            else:
                group = None
                changed = True
        else:
            # no match found, no changes required
            pass

    # Ensure requested group is present
    elif state == 'present':
        if group:
            # existing group
            if group['Description'] != description:
                module.fail_json(
                    msg="Group description does not match existing group. ec2_group does not support this case.")

        # if the group doesn't exist, create it now
        else:
            # no match found, create it
            if not module.check_mode:
                params = dict(GroupName=name, Description=description)
                if vpc_id:
                    params['VpcId'] = vpc_id
                group = client.create_security_group(**params)
                # When a group is created, an egress_rule ALLOW ALL
                # to 0.0.0.0/0 is added automatically but it's not
                # reflected in the object returned by the AWS API
                # call. We re-read the group for getting an updated object
                # amazon sometimes takes a couple seconds to update the security group so wait till it exists
                while True:
                    group = get_security_groups_with_backoff(client, GroupIds=[group['GroupId']])['SecurityGroups'][0]
                    if not group['IpPermissionsEgress']:
                        pass
                    else:
                        break

            changed = True

        if tags is not None:
            current_tags = boto3_tag_list_to_ansible_dict(group.get('Tags', []))
            tags_need_modify, tags_to_delete = compare_aws_tags(current_tags, tags, purge_tags)
            if tags_to_delete:
                try:
                    client.delete_tags(Resources=[group['GroupId']], Tags=[{'Key': tag} for tag in tags_to_delete])
                except botocore.exceptions.ClientError as e:
                    module.fail_json(msg=e.message, exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
                changed = True

            # Add/update tags
            if tags_need_modify:
                try:
                    client.create_tags(Resources=[group['GroupId']], Tags=ansible_dict_to_boto3_tag_list(tags_need_modify))
                except botocore.exceptions.ClientError as e:
                    module.fail_json(msg=e.message, exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
                changed = True

    else:
        module.fail_json(msg="Unsupported state requested: %s" % state)

    # create a lookup for all existing rules on the group
    if group:
        # Manage ingress rules
        groupRules = {}
        add_rules_to_lookup(group['IpPermissions'], group['GroupId'], 'in', groupRules)
        # Now, go through all provided rules and ensure they are there.
        if rules is not None:
            ip_permission = []
            for rule in rules:
                validate_rule(module, rule)
                group_id, ip, ipv6, target_group_created = get_target_from_rule(module, client, rule, name,
                                                                                group, groups, vpc_id)
                if target_group_created:
                    changed = True

                if rule['proto'] in ('all', '-1', -1):
                    rule['proto'] = -1
                    rule['from_port'] = None
                    rule['to_port'] = None

                if group_id:
                    rule_id = make_rule_key('in', rule, group['GroupId'], group_id)
                    if rule_id in groupRules:
                        del groupRules[rule_id]
                    else:
                        if not module.check_mode:
                            ip_permission = serialize_group_grant(group_id, rule)
                            if ip_permission:
                                ips = ip_permission
                                if vpc_id:
                                    [useridpair.update({'VpcId': vpc_id}) for useridpair in
                                     ip_permission.get('UserIdGroupPairs')]
                                try:
                                    client.authorize_security_group_ingress(GroupId=group['GroupId'], IpPermissions=[ips])
                                except botocore.exceptions.ClientError as e:
                                    module.fail_json(
                                        msg="Unable to authorize ingress for group %s security group '%s' - %s" %
                                            (group_id, group['GroupName'], e),
                                        exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
                        changed = True
                elif ip:
                    # Convert ip to list we can iterate over
                    if ip and not isinstance(ip, list):
                        ip = [ip]

                    changed, ip_permission = authorize_ip("in", changed, client, group, groupRules, ip, ip_permission,
                                                          module, rule, "ipv4")
                elif ipv6:
                    # Convert ip to list we can iterate over
                    if not isinstance(ipv6, list):
                        ipv6 = [ipv6]
                    # If rule already exists, don't later delete it
                    changed, ip_permission = authorize_ip("in", changed, client, group, groupRules, ipv6, ip_permission,
                                                          module, rule, "ipv6")
        # Finally, remove anything left in the groupRules -- these will be defunct rules
        if purge_rules:
            for (rule, grant) in groupRules.values():
                ip_permission = serialize_revoke(grant, rule)
                if not module.check_mode:
                    try:
                        client.revoke_security_group_ingress(GroupId=group['GroupId'], IpPermissions=[ip_permission])
                    except botocore.exceptions.ClientError as e:
                        module.fail_json(
                            msg="Unable to revoke ingress for security group '%s' - %s" %
                                (group['GroupName'], e),
                            exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
                changed = True

        # Manage egress rules
        groupRules = {}
        add_rules_to_lookup(group['IpPermissionsEgress'], group['GroupId'], 'out', groupRules)
        # Now, go through all provided rules and ensure they are there.
        if rules_egress is not None:
            for rule in rules_egress:
                validate_rule(module, rule)
                group_id, ip, ipv6, target_group_created = get_target_from_rule(module, client, rule, name,
                                                                                group, groups, vpc_id)
                if target_group_created:
                    changed = True

                if rule['proto'] in ('all', '-1', -1):
                    rule['proto'] = -1
                    rule['from_port'] = None
                    rule['to_port'] = None

                if group_id:
                    rule_id = make_rule_key('out', rule, group['GroupId'], group_id)
                    if rule_id in groupRules:
                        del groupRules[rule_id]
                    else:
                        if not module.check_mode:
                            ip_permission = serialize_group_grant(group_id, rule)
                            if ip_permission:
                                ips = ip_permission
                                if vpc_id:
                                    [useridpair.update({'VpcId': vpc_id}) for useridpair in
                                     ip_permission.get('UserIdGroupPairs')]
                                try:
                                    client.authorize_security_group_egress(GroupId=group['GroupId'], IpPermissions=[ips])
                                except botocore.exceptions.ClientError as e:
                                    module.fail_json(
                                        msg="Unable to authorize egress for group %s security group '%s' - %s" %
                                            (group_id, group['GroupName'], e),
                                        exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
                        changed = True
                elif ip:
                    # Convert ip to list we can iterate over
                    if not isinstance(ip, list):
                        ip = [ip]
                    changed, ip_permission = authorize_ip("out", changed, client, group, groupRules, ip,
                                                          ip_permission, module, rule, "ipv4")
                elif ipv6:
                    # Convert ip to list we can iterate over
                    if not isinstance(ipv6, list):
                        ipv6 = [ipv6]
                    # If rule already exists, don't later delete it
                    changed, ip_permission = authorize_ip("out", changed, client, group, groupRules, ipv6,
                                                          ip_permission, module, rule, "ipv6")
        else:
            # when no egress rules are specified,
            # we add in a default allow all out rule, which was the
            # default behavior before egress rules were added
            default_egress_rule = 'out--1-None-None-' + group['GroupId'] + '-0.0.0.0/0'
            if default_egress_rule not in groupRules:
                if not module.check_mode:
                    ip_permission = [{'IpProtocol': '-1',
                                      'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                                      }
                                     ]
                    try:
                        client.authorize_security_group_egress(GroupId=group['GroupId'], IpPermissions=ip_permission)
                    except botocore.exceptions.ClientError as e:
                        module.fail_json(msg="Unable to authorize egress for ip %s security group '%s' - %s" %
                                             ('0.0.0.0/0',
                                              group['GroupName'],
                                              e),
                                         exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
                changed = True
            else:
                # make sure the default egress rule is not removed
                del groupRules[default_egress_rule]

        # Finally, remove anything left in the groupRules -- these will be defunct rules
        if purge_rules_egress:
            for (rule, grant) in groupRules.values():
                # we shouldn't be revoking 0.0.0.0 egress
                if grant != '0.0.0.0/0':
                    ip_permission = serialize_revoke(grant, rule)
                    if not module.check_mode:
                        try:
                            client.revoke_security_group_egress(GroupId=group['GroupId'], IpPermissions=[ip_permission])
                        except botocore.exceptions.ClientError as e:
                            module.fail_json(msg="Unable to revoke egress for ip %s security group '%s' - %s" %
                                                 (grant, group['GroupName'], e),
                                             exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
                    changed = True

    if group:
        security_group = get_security_groups_with_backoff(client, GroupIds=[group['GroupId']])['SecurityGroups'][0]
        security_group = camel_dict_to_snake_dict(security_group)
        security_group['tags'] = boto3_tag_list_to_ansible_dict(security_group.get('tags', []),
                                                                tag_name_key_name='key', tag_value_key_name='value')
        module.exit_json(changed=changed, **security_group)
    else:
        module.exit_json(changed=changed, group_id=None)