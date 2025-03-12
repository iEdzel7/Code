def create_subnet(conn, module, vpc_id, cidr, ipv6_cidr=None, az=None, start_time=None):
    wait = module.params['wait']
    wait_timeout = module.params['wait_timeout']

    params = dict(VpcId=vpc_id,
                  CidrBlock=cidr)

    if ipv6_cidr:
        params['Ipv6CidrBlock'] = ipv6_cidr

    if az:
        params['AvailabilityZone'] = az

    try:
        subnet = get_subnet_info(conn.create_subnet(**params))
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't create subnet")

    # Sometimes AWS takes its time to create a subnet and so using
    # new subnets's id to do things like create tags results in
    # exception.
    if wait and subnet.get('state') != 'available':
        handle_waiter(conn, module, 'subnet_exists', {'SubnetIds': [subnet['id']]}, start_time)
        try:
            conn.get_waiter('subnet_available').wait(
                SubnetIds=[subnet['id']],
                WaiterConfig=wait_config(wait_timeout, start_time)
            )
            subnet['state'] = 'available'
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, "Create subnet action timed out waiting for subnet to become available")

    return subnet