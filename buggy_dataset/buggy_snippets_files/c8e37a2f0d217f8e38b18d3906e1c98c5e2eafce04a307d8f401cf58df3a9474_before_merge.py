    def _config(self):
        """Configure an LXC container.

        Write new configuration values to the lxc config file. This will
        stop the container if it's running write the new options and then
        restart the container upon completion.
        """

        _container_config = self.module.params.get('container_config')
        if not _container_config:
            return False

        container_config_file = self.container.config_file_name
        with open(container_config_file, 'rb') as f:
            container_config = f.readlines()

        # Note used ast literal_eval because AnsibleModule does not provide for
        # adequate dictionary parsing.
        # Issue: https://github.com/ansible/ansible/issues/7679
        # TODO(cloudnull) adjust import when issue has been resolved.
        import ast
        options_dict = ast.literal_eval(_container_config)
        parsed_options = [i.split('=', 1) for i in options_dict]

        config_change = False
        for key, value in parsed_options:
            key = key.strip()
            value = value.strip()
            new_entry = '%s = %s\n' % (key, value)
            keyre = re.compile(r'%s(\s+)?=' % key)
            for option_line in container_config:
                # Look for key in config
                if keyre.match(option_line):
                    _, _value = option_line.split('=', 1)
                    config_value = ' '.join(_value.split())
                    line_index = container_config.index(option_line)
                    # If the sanitized values don't match replace them
                    if value != config_value:
                        line_index += 1
                        if new_entry not in container_config:
                            config_change = True
                            container_config.insert(line_index, new_entry)
                    # Break the flow as values are written or not at this point
                    break
            else:
                config_change = True
                container_config.append(new_entry)

        # If the config changed restart the container.
        if config_change:
            container_state = self._get_state()
            if container_state != 'stopped':
                self.container.stop()

            with open(container_config_file, 'wb') as f:
                f.writelines(container_config)

            self.state_change = True
            if container_state == 'running':
                self._container_startup()
            elif container_state == 'frozen':
                self._container_startup()
                self.container.freeze()