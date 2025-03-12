def create_subnet(conn, module, vpc_id, cidr, az, check_mode):
    try:
        new_subnet = get_subnet_info(conn.create_subnet(VpcId=vpc_id, CidrBlock=cidr, AvailabilityZone=az))
        # Sometimes AWS takes its time to create a subnet and so using
        # new subnets's id to do things like create tags results in
        # exception.  boto doesn't seem to refresh 'state' of the newly
        # created subnet, i.e.: it's always 'pending'.
        subnet = False
        while subnet is False:
            subnet = subnet_exists(conn, new_subnet['id'])
            time.sleep(0.1)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "DryRunOperation":
            subnet = None
        else:
            module.fail_json(msg=e.message, exception=traceback.format_exc(),
                             **camel_dict_to_snake_dict(e.response))

    return subnet