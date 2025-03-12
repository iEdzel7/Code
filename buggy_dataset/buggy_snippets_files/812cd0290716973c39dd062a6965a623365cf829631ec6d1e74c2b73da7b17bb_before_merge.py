def create_rule_instance(rule):
    '''
    Create an instance of SecurityRule from a dict.

    :param rule: dict
    :return: SecurityRule
    '''
    return SecurityRule(
        rule['protocol'],
        rule['source_address_prefix'],
        rule['destination_address_prefix'],
        rule['access'],
        rule['direction'],
        id=rule.get('id', None),
        description=rule.get('description', None),
        source_port_range=rule.get('source_port_range', None),
        destination_port_range=rule.get('destination_port_range', None),
        priority=rule.get('priority', None),
        provisioning_state=rule.get('provisioning_state', None),
        name=rule.get('name', None),
        etag=rule.get('etag', None)
    )