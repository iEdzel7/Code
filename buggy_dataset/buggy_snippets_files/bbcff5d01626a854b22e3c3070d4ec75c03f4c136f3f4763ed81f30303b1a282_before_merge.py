    def modify_connection_vlan(self):
        cmd = [self.module.get_bin_path('nmcli', True)]
        # format for modifying ethernet interface
        return cmd