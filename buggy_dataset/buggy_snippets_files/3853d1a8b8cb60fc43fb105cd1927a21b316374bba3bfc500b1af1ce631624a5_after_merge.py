    def create_connection_ethernet(self):
        cmd = [self.nmcli_bin, 'con', 'add', 'type', 'ethernet', 'con-name']
        # format for creating ethernet interface
        # To add an Ethernet connection with static IP configuration, issue a command as follows
        # - nmcli: name=add conn_name=my-eth1 ifname=eth1 type=ethernet ip4=192.0.2.100/24 gw4=192.0.2.1 state=present
        # nmcli con add con-name my-eth1 ifname eth1 type ethernet ip4 192.0.2.100/24 gw4 192.0.2.1
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