    def group_embed_fields(fields: List[EmbedField], max_chars=1000):

        curr_group = []
        ret = []
        current_count = 0

        for i, f in enumerate(fields):
            f_len = len(f.value) + len(f.name)

            # Commands start at the 1st index of fields, i < 2 is a hacky workaround for now
            if not current_count or f_len + current_count < max_chars or i < 2:
                current_count += f_len
                curr_group.append(f)
            elif curr_group:
                ret.append(curr_group)
                current_count = f_len
                curr_group = [f]
        else:
            if curr_group:
                ret.append(curr_group)

        return ret