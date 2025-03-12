    def _parse_group(self, group, group_data):

        self.inventory.add_group(group)

        if isinstance(group_data, MutableMapping):
            # make sure they are dicts
            for section in ['vars', 'children', 'hosts']:
                if section in group_data and isinstance(group_data[section], string_types):
                    group_data[section] = {group_data[section]: None}

            if group_data.get('vars', False):
                for var in group_data['vars']:
                    self.inventory.set_variable(group, var, group_data['vars'][var])

            if group_data.get('children', False):
                for subgroup in group_data['children']:
                    self._parse_group(subgroup, group_data['children'][subgroup])
                    self.inventory.add_child(group, subgroup)

            if group_data.get('hosts', False):
                for host_pattern in group_data['hosts']:
                    hosts, port = self._parse_host(host_pattern)
                    self.populate_host_vars(hosts, group_data['hosts'][host_pattern] or {}, group, port)