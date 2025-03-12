    def get_vars(self, play=None, host=None, task=None, include_hostvars=True, include_delegate_to=True, use_cache=True):
        '''
        Returns the variables, with optional "context" given via the parameters
        for the play, host, and task (which could possibly result in different
        sets of variables being returned due to the additional context).

        The order of precedence is:
        - play->roles->get_default_vars (if there is a play context)
        - group_vars_files[host] (if there is a host context)
        - host_vars_files[host] (if there is a host context)
        - host->get_vars (if there is a host context)
        - fact_cache[host] (if there is a host context)
        - play vars (if there is a play context)
        - play vars_files (if there's no host context, ignore
          file names that cannot be templated)
        - task->get_vars (if there is a task context)
        - vars_cache[host] (if there is a host context)
        - extra vars
        '''

        display.debug("in VariableManager get_vars()")

        all_vars = dict()
        magic_variables = self._get_magic_variables(
            play=play,
            host=host,
            task=task,
            include_hostvars=include_hostvars,
            include_delegate_to=include_delegate_to,
        )

        # default for all cases
        basedirs = [self._loader.get_basedir()]

        if play:
            # first we compile any vars specified in defaults/main.yml
            # for all roles within the specified play
            for role in play.get_roles():
                all_vars = combine_vars(all_vars, role.get_default_vars())

        if task:
            # set basedirs
            if C.PLAYBOOK_VARS_ROOT == 'all':  # should be default
                basedirs = task.get_search_path()
            elif C.PLAYBOOK_VARS_ROOT in ('bottom', 'playbook_dir'):  # only option in 2.4.0
                basedirs = [task.get_search_path()[0]]
            elif C.PLAYBOOK_VARS_ROOT != 'top':
                # preserves default basedirs, only option pre 2.3
                raise AnsibleError('Unkown playbook vars logic: %s' % C.PLAYBOOK_VARS_ROOT)

            # if we have a task in this context, and that task has a role, make
            # sure it sees its defaults above any other roles, as we previously
            # (v1) made sure each task had a copy of its roles default vars
            if task._role is not None and (play or task.action == 'include_role'):
                all_vars = combine_vars(all_vars, task._role.get_default_vars(dep_chain=task.get_dep_chain()))

        if host:
            # THE 'all' group and the rest of groups for a host, used below
            all_group = self._inventory.groups.get('all')
            host_groups = sort_groups([g for g in host.get_groups() if g.name not in ['all']])

            def _get_plugin_vars(plugin, path, entities):
                data = {}
                try:
                    data = plugin.get_vars(self._loader, path, entities)
                except AttributeError:
                    try:
                        for entity in entities:
                            if isinstance(entity, Host):
                                data.update(plugin.get_host_vars(entity.name))
                            else:
                                data.update(plugin.get_group_vars(entity.name))
                    except AttributeError:
                        if hasattr(plugin, 'run'):
                            raise AnsibleError("Cannot use v1 type vars plugin %s from %s" % (plugin._load_name, plugin._original_path))
                        else:
                            raise AnsibleError("Invalid vars plugin %s from %s" % (plugin._load_name, plugin._original_path))
                return data

            # internal fuctions that actually do the work
            def _plugins_inventory(entities):
                ''' merges all entities by inventory source '''
                data = {}
                for inventory_dir in self._inventory._sources:
                    if ',' in inventory_dir and not os.path.exists(inventory_dir):  # skip host lists
                        continue
                    elif not os.path.isdir(inventory_dir):  # always pass 'inventory directory'
                        inventory_dir = os.path.dirname(inventory_dir)

                    for plugin in vars_loader.all():

                        data = combine_vars(data, _get_plugin_vars(plugin, inventory_dir, entities))
                return data

            def _plugins_play(entities):
                ''' merges all entities adjacent to play '''
                data = {}
                for plugin in vars_loader.all():

                    for path in basedirs:
                        data = combine_vars(data, _get_plugin_vars(plugin, path, entities))
                return data

            # configurable functions that are sortable via config, rememer to add to _ALLOWED if expanding this list
            def all_inventory():
                return all_group.get_vars()

            def all_plugins_inventory():
                return _plugins_inventory([all_group])

            def all_plugins_play():
                return _plugins_play([all_group])

            def groups_inventory():
                ''' gets group vars from inventory '''
                return get_group_vars(host_groups)

            def groups_plugins_inventory():
                ''' gets plugin sources from inventory for groups '''
                return _plugins_inventory(host_groups)

            def groups_plugins_play():
                ''' gets plugin sources from play for groups '''
                return _plugins_play(host_groups)

            def plugins_by_groups():
                '''
                    merges all plugin sources by group,
                    This should be used instead, NOT in combination with the other groups_plugins* functions
                '''
                data = {}
                for group in host_groups:
                    data[group] = combine_vars(data[group], _plugins_inventory(group))
                    data[group] = combine_vars(data[group], _plugins_play(group))
                return data

            # Merge groups as per precedence config
            # only allow to call the functions we want exposed
            for entry in C.VARIABLE_PRECEDENCE:
                if entry in self._ALLOWED:
                    display.debug('Calling %s to load vars for %s' % (entry, host.name))
                    all_vars = combine_vars(all_vars, locals()[entry]())
                else:
                    display.warning('Ignoring unknown variable precedence entry: %s' % (entry))

            # host vars, from inventory, inventory adjacent and play adjacent via plugins
            all_vars = combine_vars(all_vars, host.get_vars())
            all_vars = combine_vars(all_vars, _plugins_inventory([host]))
            all_vars = combine_vars(all_vars, _plugins_play([host]))

            # finally, the facts caches for this host, if it exists
            try:
                facts = self._fact_cache.get(host.name, {})
                all_vars.update(namespace_facts(facts))

                # push facts to main namespace
                if C.INJECT_FACTS_AS_VARS:
                    all_vars = combine_vars(all_vars, wrap_var(facts))
                else:
                    # always 'promote' ansible_local
                    all_vars = combine_vars(all_vars, wrap_var({'ansible_local': facts.get('ansible_local', {})}))
            except KeyError:
                pass

        if play:
            all_vars = combine_vars(all_vars, play.get_vars())

            vars_files = play.get_vars_files()
            try:
                for vars_file_item in vars_files:
                    # create a set of temporary vars here, which incorporate the extra
                    # and magic vars so we can properly template the vars_files entries
                    temp_vars = combine_vars(all_vars, self._extra_vars)
                    temp_vars = combine_vars(temp_vars, magic_variables)
                    templar = Templar(loader=self._loader, variables=temp_vars)

                    # we assume each item in the list is itself a list, as we
                    # support "conditional includes" for vars_files, which mimics
                    # the with_first_found mechanism.
                    vars_file_list = vars_file_item
                    if not isinstance(vars_file_list, list):
                        vars_file_list = [vars_file_list]

                    # now we iterate through the (potential) files, and break out
                    # as soon as we read one from the list. If none are found, we
                    # raise an error, which is silently ignored at this point.
                    try:
                        for vars_file in vars_file_list:
                            vars_file = templar.template(vars_file)
                            if not (isinstance(vars_file, Sequence)):
                                raise AnsibleError(
                                    "Invalid vars_files entry found: %r\n"
                                    "vars_files entries should be either a string type or "
                                    "a list of string types after template expansion" % vars_file
                                )
                            try:
                                data = preprocess_vars(self._loader.load_from_file(vars_file, unsafe=True))
                                if data is not None:
                                    for item in data:
                                        all_vars = combine_vars(all_vars, item)
                                break
                            except AnsibleFileNotFound:
                                # we continue on loader failures
                                continue
                            except AnsibleParserError:
                                raise
                        else:
                            # if include_delegate_to is set to False, we ignore the missing
                            # vars file here because we're working on a delegated host
                            if include_delegate_to:
                                raise AnsibleFileNotFound("vars file %s was not found" % vars_file_item)
                    except (UndefinedError, AnsibleUndefinedVariable):
                        if host is not None and self._fact_cache.get(host.name, dict()).get('module_setup') and task is not None:
                            raise AnsibleUndefinedVariable("an undefined variable was found when attempting to template the vars_files item '%s'"
                                                           % vars_file_item, obj=vars_file_item)
                        else:
                            # we do not have a full context here, and the missing variable could be because of that
                            # so just show a warning and continue
                            display.vvv("skipping vars_file '%s' due to an undefined variable" % vars_file_item)
                            continue

                    display.vvv("Read vars_file '%s'" % vars_file_item)
            except TypeError:
                raise AnsibleParserError("Error while reading vars files - please supply a list of file names. "
                                         "Got '%s' of type %s" % (vars_files, type(vars_files)))

            # By default, we now merge in all vars from all roles in the play,
            # unless the user has disabled this via a config option
            if not C.DEFAULT_PRIVATE_ROLE_VARS:
                for role in play.get_roles():
                    all_vars = combine_vars(all_vars, role.get_vars(include_params=False))

        # next, we merge in the vars from the role, which will specifically
        # follow the role dependency chain, and then we merge in the tasks
        # vars (which will look at parent blocks/task includes)
        if task:
            if task._role:
                all_vars = combine_vars(all_vars, task._role.get_vars(task.get_dep_chain(), include_params=False))
            all_vars = combine_vars(all_vars, task.get_vars())

        # next, we merge in the vars cache (include vars) and nonpersistent
        # facts cache (set_fact/register), in that order
        if host:
            # include_vars non-persistent cache
            all_vars = combine_vars(all_vars, self._vars_cache.get(host.get_name(), dict()))
            # fact non-persistent cache
            all_vars = combine_vars(all_vars, self._nonpersistent_fact_cache.get(host.name, dict()))

        # next, we merge in role params and task include params
        if task:
            if task._role:
                all_vars = combine_vars(all_vars, task._role.get_role_params(task.get_dep_chain()))

            # special case for include tasks, where the include params
            # may be specified in the vars field for the task, which should
            # have higher precedence than the vars/np facts above
            all_vars = combine_vars(all_vars, task.get_include_params())

        # extra vars
        all_vars = combine_vars(all_vars, self._extra_vars)

        # magic variables
        all_vars = combine_vars(all_vars, magic_variables)

        # special case for the 'environment' magic variable, as someone
        # may have set it as a variable and we don't want to stomp on it
        if task:
            all_vars['environment'] = task.environment

        # if we have a task and we're delegating to another host, figure out the
        # variables for that host now so we don't have to rely on hostvars later
        if task and task.delegate_to is not None and include_delegate_to:
            all_vars['ansible_delegated_vars'] = self._get_delegated_vars(play, task, all_vars)

        # 'vars' magic var
        if task or play:
            # has to be copy, otherwise recursive ref
            all_vars['vars'] = all_vars.copy()

        display.debug("done with get_vars()")
        return all_vars