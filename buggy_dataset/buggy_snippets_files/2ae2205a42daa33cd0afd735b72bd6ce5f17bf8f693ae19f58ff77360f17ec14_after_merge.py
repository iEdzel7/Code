    def parse_version(self, data):
        match = re.search(r'Version (\S+?)(?:,\s|\s)', data)
        if match:
            return match.group(1)