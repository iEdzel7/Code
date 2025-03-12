    def parse_module(self, data):
        objects = list()
        for line in data.splitlines():
            if line == '' or re.search(r'\b' + 'Sw' + r'\b', line):
                break
            if line[0].isdigit():
                obj = {}
                match_port = re.search(r'\d\s*(\d*)', line, re.M)
                if match_port:
                    obj['ports'] = match_port.group(1)

                match = re.search(r'\d\s*\d*\s*(.+)$', line, re.M)
                if match:
                    l = match.group(1).split('  ')
                    items = list()
                    for item in l:
                        if item == '':
                            continue
                        items.append(item.strip())

                    if items:
                        obj['type'] = items[0]
                        obj['model'] = items[1]
                        obj['status'] = items[-1]

                objects.append(obj)
        return objects