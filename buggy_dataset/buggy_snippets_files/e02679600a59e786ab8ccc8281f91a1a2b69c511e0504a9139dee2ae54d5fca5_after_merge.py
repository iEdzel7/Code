def create_autoscaling_group(connection, module):
    group_name = module.params.get('name')
    load_balancers = module.params['load_balancers']
    target_group_arns = module.params['target_group_arns']
    availability_zones = module.params['availability_zones']
    launch_config_name = module.params.get('launch_config_name')
    min_size = module.params['min_size']
    max_size = module.params['max_size']
    placement_group = module.params.get('placement_group')
    desired_capacity = module.params.get('desired_capacity')
    vpc_zone_identifier = module.params.get('vpc_zone_identifier')
    set_tags = module.params.get('tags')
    health_check_period = module.params.get('health_check_period')
    health_check_type = module.params.get('health_check_type')
    default_cooldown = module.params.get('default_cooldown')
    wait_for_instances = module.params.get('wait_for_instances')
    as_groups = connection.describe_auto_scaling_groups(AutoScalingGroupNames=[group_name])
    wait_timeout = module.params.get('wait_timeout')
    termination_policies = module.params.get('termination_policies')
    notification_topic = module.params.get('notification_topic')
    notification_types = module.params.get('notification_types')

    if not vpc_zone_identifier and not availability_zones:
        region, ec2_url, aws_connect_params = get_aws_connection_info(module, boto3=True)
        ec2_connection = boto3_conn(module,
                                    conn_type='client',
                                    resource='ec2',
                                    region=region,
                                    endpoint=ec2_url,
                                    **aws_connect_params)
    elif vpc_zone_identifier:
        vpc_zone_identifier = ','.join(vpc_zone_identifier)

    asg_tags = []
    for tag in set_tags:
        for k, v in tag.items():
            if k != 'propagate_at_launch':
                asg_tags.append(dict(Key=k,
                                     Value=v,
                                     PropagateAtLaunch=bool(tag.get('propagate_at_launch', True)),
                                     ResourceType='auto-scaling-group',
                                     ResourceId=group_name))
    if not as_groups.get('AutoScalingGroups'):
        if not vpc_zone_identifier and not availability_zones:
            availability_zones = module.params['availability_zones'] = [zone['ZoneName'] for
                                                                        zone in ec2_connection.describe_availability_zones()['AvailabilityZones']]
        enforce_required_arguments(module)
        launch_configs = describe_launch_configurations(connection, launch_config_name)
        if len(launch_configs['LaunchConfigurations']) == 0:
            module.fail_json(msg="No launch config found with name %s" % launch_config_name)
        ag = dict(
            AutoScalingGroupName=group_name,
            LaunchConfigurationName=launch_configs['LaunchConfigurations'][0]['LaunchConfigurationName'],
            MinSize=min_size,
            MaxSize=max_size,
            DesiredCapacity=desired_capacity,
            Tags=asg_tags,
            HealthCheckGracePeriod=health_check_period,
            HealthCheckType=health_check_type,
            DefaultCooldown=default_cooldown,
            TerminationPolicies=termination_policies)
        if vpc_zone_identifier:
            ag['VPCZoneIdentifier'] = vpc_zone_identifier
        if availability_zones:
            ag['AvailabilityZones'] = availability_zones
        if placement_group:
            ag['PlacementGroup'] = placement_group
        if load_balancers:
            ag['LoadBalancerNames'] = load_balancers
        if target_group_arns:
            ag['TargetGroupARNs'] = target_group_arns

        try:
            create_asg(connection, **ag)

            all_ag = describe_autoscaling_groups(connection, group_name)
            if len(all_ag) == 0:
                module.fail_json(msg="No auto scaling group found with the name %s" % group_name)
            as_group = all_ag[0]
            suspend_processes(connection, as_group, module)
            if wait_for_instances:
                wait_for_new_inst(module, connection, group_name, wait_timeout, desired_capacity, 'viable_instances')
                if load_balancers:
                    wait_for_elb(connection, module, group_name)
                # Wait for target group health if target group(s)defined
                if target_group_arns:
                    wait_for_target_group(connection, module, group_name)
            if notification_topic:
                put_notification_config(connection, group_name, notification_topic, notification_types)
            as_group = describe_autoscaling_groups(connection, group_name)[0]
            asg_properties = get_properties(as_group, module)
            changed = True
            return changed, asg_properties
        except botocore.exceptions.ClientError as e:
            module.fail_json(msg="Failed to create Autoscaling Group.",
                             exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
        except botocore.exceptions.BotoCoreError as e:
            module.fail_json(msg="Failed to create Autoscaling Group.",
                             exception=traceback.format_exc())
    else:
        as_group = as_groups['AutoScalingGroups'][0]
        initial_asg_properties = get_properties(as_group, module)
        changed = False

        if suspend_processes(connection, as_group, module):
            changed = True

        # process tag changes
        if len(set_tags) > 0:
            have_tags = as_group.get('Tags')
            want_tags = asg_tags
            dead_tags = []
            have_tag_keyvals = [x['Key'] for x in have_tags]
            want_tag_keyvals = [x['Key'] for x in want_tags]

            for dead_tag in set(have_tag_keyvals).difference(want_tag_keyvals):
                changed = True
                dead_tags.append(dict(ResourceId=as_group['AutoScalingGroupName'],
                                      ResourceType='auto-scaling-group', Key=dead_tag))
                have_tags = [have_tag for have_tag in have_tags if have_tag['Key'] != dead_tag]
            if dead_tags:
                connection.delete_tags(Tags=dead_tags)

            zipped = zip(have_tags, want_tags)
            if len(have_tags) != len(want_tags) or not all(x == y for x, y in zipped):
                changed = True
                connection.create_or_update_tags(Tags=asg_tags)

        # Handle load balancer attachments/detachments
        # Attach load balancers if they are specified but none currently exist
        if load_balancers and not as_group['LoadBalancerNames']:
            changed = True
            try:
                attach_load_balancers(connection, group_name, load_balancers)
            except botocore.exceptions.ClientError as e:
                module.fail_json(msg="Failed to update Autoscaling Group.",
                                 exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
            except botocore.exceptions.BotoCoreError as e:
                module.fail_json(msg="Failed to update Autoscaling Group.",
                                 exception=traceback.format_exc())

        # Update load balancers if they are specified and one or more already exists
        elif as_group['LoadBalancerNames']:
            change_load_balancers = load_balancers is not None
            # Get differences
            if not load_balancers:
                load_balancers = list()
            wanted_elbs = set(load_balancers)

            has_elbs = set(as_group['LoadBalancerNames'])
            # check if all requested are already existing
            if has_elbs - wanted_elbs and change_load_balancers:
                # if wanted contains less than existing, then we need to delete some
                elbs_to_detach = has_elbs.difference(wanted_elbs)
                if elbs_to_detach:
                    changed = True
                    detach_load_balancers(connection, group_name, list(elbs_to_detach))
            if wanted_elbs - has_elbs:
                # if has contains less than wanted, then we need to add some
                elbs_to_attach = wanted_elbs.difference(has_elbs)
                if elbs_to_attach:
                    changed = True
                    attach_load_balancers(connection, group_name, list(elbs_to_attach))

        # Handle target group attachments/detachments
        # Attach target groups if they are specified but none currently exist
        if target_group_arns and not as_group['TargetGroupARNs']:
            changed = True
            try:
                attach_lb_target_groups(connection, group_name, target_group_arns)
            except botocore.exceptions.ClientError as e:
                module.fail_json(msg="Failed to update Autoscaling Group.",
                                 exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
            except botocore.exceptions.BotoCoreError as e:
                module.fail_json(msg="Failed to update Autoscaling Group.",
                                 exception=traceback.format_exc())
        # Update target groups if they are specified and one or more already exists
        elif target_group_arns is not None and as_group['TargetGroupARNs']:
            # Get differences
            wanted_tgs = set(target_group_arns)
            has_tgs = set(as_group['TargetGroupARNs'])
            # check if all requested are already existing
            if has_tgs.issuperset(wanted_tgs):
                # if wanted contains less than existing, then we need to delete some
                tgs_to_detach = has_tgs.difference(wanted_tgs)
                if tgs_to_detach:
                    changed = True
                    detach_lb_target_groups(connection, group_name, tgs_to_detach)
            if wanted_tgs.issuperset(has_tgs):
                # if has contains less than wanted, then we need to add some
                tgs_to_attach = wanted_tgs.difference(has_tgs)
                if tgs_to_attach:
                    changed = True
                    attach_lb_target_groups(connection, group_name, tgs_to_attach)

        # check for attributes that aren't required for updating an existing ASG
        desired_capacity = desired_capacity if desired_capacity is not None else as_group['DesiredCapacity']
        min_size = min_size if min_size is not None else as_group['MinSize']
        max_size = max_size if max_size is not None else as_group['MaxSize']
        launch_config_name = launch_config_name or as_group['LaunchConfigurationName']

        launch_configs = describe_launch_configurations(connection, launch_config_name)
        if len(launch_configs['LaunchConfigurations']) == 0:
            module.fail_json(msg="No launch config found with name %s" % launch_config_name)
        ag = dict(
            AutoScalingGroupName=group_name,
            LaunchConfigurationName=launch_configs['LaunchConfigurations'][0]['LaunchConfigurationName'],
            MinSize=min_size,
            MaxSize=max_size,
            DesiredCapacity=desired_capacity,
            HealthCheckGracePeriod=health_check_period,
            HealthCheckType=health_check_type,
            DefaultCooldown=default_cooldown,
            TerminationPolicies=termination_policies)
        if availability_zones:
            ag['AvailabilityZones'] = availability_zones
        if vpc_zone_identifier:
            ag['VPCZoneIdentifier'] = vpc_zone_identifier
        update_asg(connection, **ag)

        if notification_topic:
            try:
                put_notification_config(connection, group_name, notification_topic, notification_types)
            except botocore.exceptions.ClientError as e:
                module.fail_json(msg="Failed to update Autoscaling Group notifications.",
                                 exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
            except botocore.exceptions.BotoCoreError as e:
                module.fail_json(msg="Failed to update Autoscaling Group notifications.",
                                 exception=traceback.format_exc())
        if wait_for_instances:
            wait_for_new_inst(module, connection, group_name, wait_timeout, desired_capacity, 'viable_instances')
            # Wait for ELB health if ELB(s)defined
            if load_balancers:
                log.debug('\tWAITING FOR ELB HEALTH')
                wait_for_elb(connection, module, group_name)
            # Wait for target group health if target group(s)defined

            if target_group_arns:
                log.debug('\tWAITING FOR TG HEALTH')
                wait_for_target_group(connection, module, group_name)

        try:
            as_group = describe_autoscaling_groups(connection, group_name)[0]
            asg_properties = get_properties(as_group, module)
            if asg_properties != initial_asg_properties:
                changed = True
        except botocore.exceptions.ClientError as e:
            module.fail_json(msg="Failed to read existing Autoscaling Groups.",
                             exception=traceback.format_exc(), **camel_dict_to_snake_dict(e.response))
        except botocore.exceptions.BotoCoreError as e:
            module.fail_json(msg="Failed to read existing Autoscaling Groups.",
                             exception=traceback.format_exc())
        return changed, asg_properties