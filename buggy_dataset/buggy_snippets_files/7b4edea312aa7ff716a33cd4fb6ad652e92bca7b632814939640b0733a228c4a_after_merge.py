def copy_image(module, ec2):
    """
    Copies an AMI

    module : AnsibleModule object
    ec2: ec2 connection object
    """

    params = {'SourceRegion': module.params.get('source_region'),
              'SourceImageId': module.params.get('source_image_id'),
              'Name': module.params.get('name'),
              'Description': module.params.get('description'),
              'Encrypted': module.params.get('encrypted'),
              }
    if module.params.get('kms_key_id'):
        params['KmsKeyId'] = module.params.get('kms_key_id')

    try:
        image_id = ec2.copy_image(**params)['ImageId']
        if module.params.get('wait'):
            delay = 15
            max_attempts = module.params.get('wait_timeout') // delay
            ec2.get_waiter('image_available').wait(
                ImageIds=[image_id],
                WaiterConfig={'Delay': delay, 'MaxAttempts': max_attempts}
            )
        if module.params.get('tags'):
            ec2.create_tags(
                Resources=[image_id],
                Tags=[{'Key': k, 'Value': v} for k, v in module.params.get('tags').items()]
            )

        module.exit_json(changed=True, image_id=image_id)
    except WaiterError as we:
        module.fail_json(msg='An error occurred waiting for the image to become available. (%s)' % str(we), exception=traceback.format_exc())
    except ClientError as ce:
        module.fail_json(msg=ce.message)
    except NoCredentialsError:
        module.fail_json(msg='Unable to authenticate, AWS credentials are invalid.')
    except Exception as e:
        module.fail_json(msg='Unhandled exception. (%s)' % str(e))