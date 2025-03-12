    def has_update(self, nic_service):
        update = False
        bond = self._module.params['bond']
        networks = self._module.params['networks']
        labels = self._module.params['labels']
        nic = get_entity(nic_service)

        if nic is None:
            return update

        # Check if bond configuration should be updated:
        if bond:
            update = self.__compare_options(get_mode_type(bond.get('mode')), getattr(nic.bonding, 'options', []))
            update = update or not equal(
                sorted(bond.get('interfaces')) if bond.get('interfaces') else None,
                sorted(get_link_name(self._connection, s) for s in nic.bonding.slaves)
            )

        # Check if labels need to be updated on interface/bond:
        if labels:
            net_labels = nic_service.network_labels_service().list()
            # If any lables which user passed aren't assigned, relabel the interface:
            if sorted(labels) != sorted([lbl.id for lbl in net_labels]):
                return True

        if not networks:
            return update

        # Check if networks attachments configuration should be updated:
        attachments_service = nic_service.network_attachments_service()
        network_names = [network.get('name') for network in networks]

        attachments = {}
        for attachment in attachments_service.list():
            name = get_link_name(self._connection, attachment.network)
            if name in network_names:
                attachments[name] = attachment

        for network in networks:
            attachment = attachments.get(network.get('name'))
            # If attachment don't exsits, we need to create it:
            if attachment is None:
                return True

            self.update_address(attachments_service, attachment, network)

        return update