def update_stack(module, stack_params, cfn):
    if 'TemplateBody' not in stack_params and 'TemplateURL' not in stack_params:
        stack_params['UsePreviousTemplate'] = True

    # if the state is present and the stack already exists, we try to update it.
    # AWS will tell us if the stack template and parameters are the same and
    # don't need to be updated.
    try:
        cfn.update_stack(**stack_params)
        result = stack_operation(cfn, stack_params['StackName'], 'UPDATE', stack_params['ClientRequestToken'])
    except Exception as err:
        error_msg = boto_exception(err)
        if 'No updates are to be performed.' in error_msg:
            result = dict(changed=False, output='Stack is already up-to-date.')
        else:
            module.fail_json(msg="Failed to update stack {0}: {1}".format(stack_params.get('StackName'), error_msg), exception=traceback.format_exc())
    if not result:
        module.fail_json(msg="empty result")
    return result