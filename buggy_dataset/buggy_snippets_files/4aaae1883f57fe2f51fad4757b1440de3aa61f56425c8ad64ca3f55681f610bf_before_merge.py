    def get_bios_attributes(self):
        result = {}
        bios_attributes = {}
        key = "Bios"

        # Search for 'key' entry and extract URI from it
        response = self.get_request(self.root_uri + self.systems_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        bios_uri = data[key]["@odata.id"]

        response = self.get_request(self.root_uri + bios_uri)
        if response['ret'] is False:
            return response
        result['ret'] = True
        data = response['data']
        for attribute in data[u'Attributes'].items():
            bios_attributes[attribute[0]] = attribute[1]
        result["entries"] = bios_attributes
        return result