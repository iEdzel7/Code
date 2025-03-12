    def __compare_options(self, new_options, old_options):
        return sorted((get_dict_of_struct(opt) for opt in new_options),
                      key=lambda x: x["name"]) != sorted((get_dict_of_struct(opt) for opt in old_options),
                                                         key=lambda x: x["name"])