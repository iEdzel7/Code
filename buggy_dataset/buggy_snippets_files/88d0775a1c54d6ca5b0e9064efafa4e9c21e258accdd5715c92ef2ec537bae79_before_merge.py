def get_vpc(module, connection, vpc_id):
    try:
        vpc_obj = AWSRetry.backoff(
            delay=1, tries=5,
            catch_extra_error_codes=['InvalidVpcID.NotFound'],
        )(connection.describe_vpcs)(VpcIds=[vpc_id])['Vpcs'][0]
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Failed to describe VPCs")
    try:
        classic_link = connection.describe_vpc_classic_link(VpcIds=[vpc_id])['Vpcs'][0].get('ClassicLinkEnabled')
        vpc_obj['ClassicLinkEnabled'] = classic_link
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Message"] == "The functionality you requested is not available in this region.":
            vpc_obj['ClassicLinkEnabled'] = False
        else:
            module.fail_json_aws(e, msg="Failed to describe VPCs")
    except botocore.exceptions.BotoCoreError as e:
        module.fail_json_aws(e, msg="Failed to describe VPCs")

    return vpc_obj