    def get_bios_boot_order(self):
        result = {}
        boot_device_list = []
        boot_device_details = []
        key = "Bios"
        bootsources = "BootSources"
        # Get these entries, but does not fail if not found
        properties = ['Index', 'Id', 'Name', 'Enabled']

        # Search for 'key' entry and extract URI from it
        response = self.get_request(self.root_uri + self.systems_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']

        if key not in data:
            return {'ret': False, 'msg': "Key %s not found" % key}

        bios_uri = data[key]["@odata.id"]

        # Get boot mode first as it will determine what attribute to read
        response = self.get_request(self.root_uri + bios_uri)
        if response['ret'] is False:
            return response
        data = response['data']
        boot_mode = data[u'Attributes']["BootMode"]
        if boot_mode == "Uefi":
            boot_seq = "UefiBootSeq"
        else:
            boot_seq = "BootSeq"

        response = self.get_request(self.root_uri + self.systems_uri + "/" + bootsources)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']

        boot_device_list = data[u'Attributes'][boot_seq]
        for b in boot_device_list:
            boot_device = {}
            for property in properties:
                if property in b:
                    boot_device[property] = b[property]
            boot_device_details.append(boot_device)
        result["entries"] = boot_device_details
        return result