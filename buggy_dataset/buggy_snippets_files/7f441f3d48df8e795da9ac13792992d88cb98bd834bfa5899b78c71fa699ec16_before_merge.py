    def populate_filesystems(self):
        data = self.responses[0]
        fs = re.findall(r'^Directory of (.+)/', data, re.M)
        return dict(filesystems=fs)