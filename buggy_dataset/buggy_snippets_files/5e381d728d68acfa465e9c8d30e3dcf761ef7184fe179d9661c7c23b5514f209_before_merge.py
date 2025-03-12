def construct_rule(params):
    rule = []
    append_param(rule, params['protocol'], '-p', False)
    append_param(rule, params['source'], '-s', False)
    append_param(rule, params['destination'], '-d', False)
    append_param(rule, params['match'], '-m', True)
    append_param(rule, params['jump'], '-j', False)
    append_param(rule, params['to_destination'], '--to-destination', False)
    append_param(rule, params['to_source'], '--to-source', False)
    append_param(rule, params['goto'], '-g', False)
    append_param(rule, params['in_interface'], '-i', False)
    append_param(rule, params['out_interface'], '-o', False)
    append_param(rule, params['fragment'], '-f', False)
    append_param(rule, params['set_counters'], '-c', False)
    append_param(rule, params['source_port'], '--source-port', False)
    append_param(rule, params['destination_port'], '--destination-port', False)
    append_param(rule, params['to_ports'], '--to-ports', False)
    append_param(rule, params['set_dscp_mark'], '--set-dscp', False)
    append_param(
        rule,
        params['set_dscp_mark_class'],
        '--set-dscp-class',
        False)
    append_match(rule, params['comment'], 'comment')
    append_param(rule, params['comment'], '--comment', False)
    if 'conntrack' in params['match']:
        append_csv(rule, params['ctstate'], '--ctstate')
    elif 'state' in params['match']:
        append_csv(rule, params['ctstate'], '--state')
    elif params['ctstate']:
        append_match(rule, params['ctstate'], 'conntrack')
        append_csv(rule, params['ctstate'], '--ctstate')
    else:
        return False
    append_match(rule, params['limit'] or params['limit_burst'], 'limit')
    append_param(rule, params['limit'], '--limit', False)
    append_param(rule, params['limit_burst'], '--limit-burst', False)
    append_match(rule, params['uid_owner'], 'owner')
    append_param(rule, params['uid_owner'], '--uid-owner', False)
    append_jump(rule, params['reject_with'], 'REJECT')
    append_param(rule, params['reject_with'], '--reject-with', False)
    append_param(
        rule,
        params['icmp_type'],
        ICMP_TYPE_OPTIONS[params['ip_version']],
        False)
    return rule