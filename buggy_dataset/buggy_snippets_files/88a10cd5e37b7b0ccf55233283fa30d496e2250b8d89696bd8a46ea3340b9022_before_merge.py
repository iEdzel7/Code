    def up_connection(self):
        cmd = [self.module.get_bin_path('nmcli', True)]
        cmd.append('con')
        cmd.append('up')
        cmd.append(self.conn_name)
        return self.execute_command(cmd)