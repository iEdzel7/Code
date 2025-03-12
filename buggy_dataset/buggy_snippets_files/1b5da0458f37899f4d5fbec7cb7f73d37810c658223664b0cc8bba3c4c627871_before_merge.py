def make_current_modifiable_param_dict(module, conn, name):
    """ Gets the current state of the cache parameter group and creates a dict with the format: {ParameterName: [Allowed_Values, DataType, ParameterValue]}"""
    current_info = get_info(conn, name)
    if current_info is False:
        module.fail_json(msg="Could not connect to the cache parameter group %s." % name)

    parameters = current_info["Parameters"]
    modifiable_params = {}

    for param in parameters:
        if param["IsModifiable"] and ("AllowedValues" and "ParameterValue") in param:
            modifiable_params[param["ParameterName"]] = [param["AllowedValues"], param["DataType"], param["ParameterValue"]]
    return modifiable_params