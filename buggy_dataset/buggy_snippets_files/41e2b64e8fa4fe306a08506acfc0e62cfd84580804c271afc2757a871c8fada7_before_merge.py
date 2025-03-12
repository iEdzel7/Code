    def modify_connection_ethernet(self):
        cmd = [self.module.get_bin_path('nmcli', True)]
        # format for  modifying ethernet interface
        # To add an Ethernet connection with static IP configuration, issue a command as follows
        # - nmcli: name=add conn_name=my-eth1 ifname=eth1 type=ethernet ip4=192.0.2.100/24 gw4=192.0.2.1 state=present
        # nmcli con add con-name my-eth1 ifname eth1 type ethernet ip4 192.0.2.100/24 gw4 192.0.2.1
        cmd.append('con')
        cmd.append('mod')
        cmd.append(self.conn_name)
        if self.ip4 is not None:
            cmd.append('ipv4.address')
            cmd.append(self.ip4)
        if self.gw4 is not None:
            cmd.append('ipv4.gateway')
            cmd.append(self.gw4)
        if self.dns4 is not None:
            cmd.append('ipv4.dns')
            cmd.append(self.dns4)
        if self.ip6 is not None:
            cmd.append('ipv6.address')
            cmd.append(self.ip6)
        if self.gw6 is not None:
            cmd.append('ipv6.gateway')
            cmd.append(self.gw6)
        if self.dns6 is not None:
            cmd.append('ipv6.dns')
            cmd.append(self.dns6)
        if self.mtu is not None:
            cmd.append('802-3-ethernet.mtu')
            cmd.append(self.mtu)
        if self.autoconnect is not None:
            cmd.append('autoconnect')
            cmd.append(self.bool_to_string(self.autoconnect))
        return cmd