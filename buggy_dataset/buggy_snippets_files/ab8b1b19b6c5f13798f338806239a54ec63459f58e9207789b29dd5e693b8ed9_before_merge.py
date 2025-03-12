def main():
    spec = dict(
        gather_subset=dict(default=['!config'], type='list')
    )

    spec.update(iosxr_argument_spec)

    module = AnsibleModule(argument_spec=spec,
                           supports_check_mode=True)

    warnings = list()

    gather_subset = module.params['gather_subset']

    runable_subsets = set()
    exclude_subsets = set()

    for subset in gather_subset:
        if subset == 'all':
            runable_subsets.update(VALID_SUBSETS)
            continue

        if subset.startswith('!'):
            subset = subset[1:]
            if subset == 'all':
                exclude_subsets.update(VALID_SUBSETS)
                continue
            exclude = True
        else:
            exclude = False

        if subset not in VALID_SUBSETS:
            module.fail_json(msg='Bad subset')

        if exclude:
            exclude_subsets.add(subset)
        else:
            runable_subsets.add(subset)

    if not runable_subsets:
        runable_subsets.update(VALID_SUBSETS)

    runable_subsets.difference_update(exclude_subsets)
    runable_subsets.add('default')

    facts = dict()
    facts['gather_subset'] = list(runable_subsets)

    instances = list()
    for key in runable_subsets:
        instances.append(FACT_SUBSETS[key]())

    try:
        for inst in instances:
            commands = inst.commands()
            responses = run_command(module, commands)
            results = dict(zip(commands, responses))
            inst.populate(results)
            facts.update(inst.facts)
    except Exception:
        module.exit_json(out=module.from_json(results))

    ansible_facts = dict()
    for key, value in iteritems(facts):
        key = 'ansible_net_%s' % key
        ansible_facts[key] = value

    module.exit_json(ansible_facts=ansible_facts, warnings=warnings)