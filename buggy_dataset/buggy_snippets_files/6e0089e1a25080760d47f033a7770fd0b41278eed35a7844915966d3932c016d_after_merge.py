    def create_connection_bond_slave(self):
        cmd = [self.nmcli_bin, 'connection', 'add', 'type', 'bond-slave', 'con-name']
        # format for creating bond-slave interface
        if self.conn_name is not None:
            cmd.append(self.conn_name)
        elif self.ifname is not None:
            cmd.append(self.ifname)
        cmd.append('ifname')
        if self.ifname is not None:
            cmd.append(self.ifname)
        elif self.conn_name is not None:
            cmd.append(self.conn_name)
        cmd.append('master')
        if self.conn_name is not None:
            cmd.append(self.master)
        return cmd