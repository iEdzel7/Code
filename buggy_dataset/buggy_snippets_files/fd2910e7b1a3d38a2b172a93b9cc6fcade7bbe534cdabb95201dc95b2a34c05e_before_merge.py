    def parse(self, inventory, loader, path, cache=True):
        super(InventoryModule, self).parse(inventory, loader, path)
        self._read_config_data(path=path)

        # Netbox access
        token = self.get_option("token")
        self.api_endpoint = self.get_option("api_endpoint")
        self.timeout = self.get_option("timeout")
        self.headers = {
            'Authorization': "Token %s" % token,
            'User-Agent': "ansible %s Python %s" % (ansible_version, python_version.split(' ')[0]),
            'Content-type': 'application/json'
        }

        # Filter and group_by options
        self.group_by = self.get_option("group_by")
        self.query_filters = self.get_option("query_filters")
        self.main()