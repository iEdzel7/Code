    def create_connection_team(self):
        cmd = [self.module.get_bin_path('nmcli', True)]
        # format for creating team interface
        cmd.append('con')
        cmd.append('add')
        cmd.append('type')
        cmd.append('team')
        cmd.append('con-name')
        if self.conn_name is not None:
            cmd.append(self.conn_name)
        elif self.ifname is not None:
            cmd.append(self.ifname)
        cmd.append('ifname')
        if self.ifname is not None:
            cmd.append(self.ifname)
        elif self.conn_name is not None:
            cmd.append(self.conn_name)
        if self.ip4 is not None:
            cmd.append('ip4')
            cmd.append(self.ip4)
        if self.gw4 is not None:
            cmd.append('gw4')
            cmd.append(self.gw4)
        if self.ip6 is not None:
            cmd.append('ip6')
            cmd.append(self.ip6)
        if self.gw6 is not None:
            cmd.append('gw6')
            cmd.append(self.gw6)
        if self.autoconnect is not None:
            cmd.append('autoconnect')
            cmd.append(self.bool_to_string(self.autoconnect))
        return cmd