    def create_default_securitygroup(self, resource_group, location, security_group_name, os_type, open_ports):
        '''
        Create a default security group <security_group_name> to associate with a network interface. If a security group matching
        <security_group_name> exists, return it. Otherwise, create one.

        :param resource_group: Resource group name
        :param location: azure location name
        :param security_group_name: base name to use for the security group
        :param os_type: one of 'Windows' or 'Linux'. Determins any default rules added to the security group.
        :param ssh_port: for os_type 'Linux' port used in rule allowing SSH access.
        :param rdp_port: for os_type 'Windows' port used in rule allowing RDP access.
        :return: security_group object
        '''
        group = None

        self.log("Create security group {0}".format(security_group_name))
        self.log("Check to see if security group {0} exists".format(security_group_name))
        try:
            group = self.network_client.network_security_groups.get(resource_group, security_group_name)
        except CloudError:
            pass

        if group:
            self.log("Security group {0} found.".format(security_group_name))
            self.check_provisioning_state(group)
            return group

        parameters = self.network_models.NetworkSecurityGroup()
        parameters.location = location

        if not open_ports:
            # Open default ports based on OS type
            if os_type == 'Linux':
                # add an inbound SSH rule
                parameters.security_rules = [
                    self.network_models.SecurityRule(protocol='Tcp',
                                                     source_address_prefix='*',
                                                     destination_address_prefix='*',
                                                     access='Allow',
                                                     direction='Inbound',
                                                     description='Allow SSH Access',
                                                     source_port_range='*',
                                                     destination_port_range='22',
                                                     priority=100,
                                                     name='SSH')
                ]
                parameters.location = location
            else:
                # for windows add inbound RDP and WinRM rules
                parameters.security_rules = [
                    self.network_models.SecurityRule(protocol='Tcp',
                                                     source_address_prefix='*',
                                                     destination_address_prefix='*',
                                                     access='Allow',
                                                     direction='Inbound',
                                                     description='Allow RDP port 3389',
                                                     source_port_range='*',
                                                     destination_port_range='3389',
                                                     priority=100,
                                                     name='RDP01'),
                    self.network_models.SecurityRule(protocol='Tcp',
                                                     source_address_prefix='*',
                                                     destination_address_prefix='*',
                                                     access='Allow',
                                                     direction='Inbound',
                                                     description='Allow WinRM HTTPS port 5986',
                                                     source_port_range='*',
                                                     destination_port_range='5986',
                                                     priority=101,
                                                     name='WinRM01'),
                ]
        else:
            # Open custom ports
            parameters.security_rules = []
            priority = 100
            for port in open_ports:
                priority += 1
                rule_name = "Rule_{0}".format(priority)
                parameters.security_rules.append(
                    self.network_models.SecurityRule('Tcp', '*', '*', 'Allow', 'Inbound', source_port_range='*',
                                                     destination_port_range=str(port), priority=priority, name=rule_name)
                )

        self.log('Creating default security group {0}'.format(security_group_name))
        try:
            poller = self.network_client.network_security_groups.create_or_update(resource_group,
                                                                                  security_group_name,
                                                                                  parameters)
        except Exception as exc:
            self.fail("Error creating default security rule {0} - {1}".format(security_group_name, str(exc)))

        return self.get_poller_result(poller)