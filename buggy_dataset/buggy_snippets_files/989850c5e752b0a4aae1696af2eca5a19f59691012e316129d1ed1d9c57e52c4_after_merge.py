    def update_address(self, attachments_service, attachment, network):
        # Check if there is any change in address assignenmts and
        # update it if needed:
        for ip in attachment.ip_address_assignments:
            if str(ip.ip.version) == network.get('version', 'v4'):
                changed = False
                if not equal(network.get('boot_protocol'), str(ip.assignment_method)):
                    ip.assignment_method = otypes.BootProtocol(network.get('boot_protocol'))
                    changed = True
                if not equal(network.get('address'), ip.ip.address):
                    ip.ip.address = network.get('address')
                    changed = True
                if not equal(network.get('gateway'), ip.ip.gateway):
                    ip.ip.gateway = network.get('gateway')
                    changed = True
                if not equal(network.get('netmask'), ip.ip.netmask):
                    ip.ip.netmask = network.get('netmask')
                    changed = True

                if changed:
                    if not self._module.check_mode:
                        attachments_service.service(attachment.id).update(attachment)
                    self.changed = True
                    break