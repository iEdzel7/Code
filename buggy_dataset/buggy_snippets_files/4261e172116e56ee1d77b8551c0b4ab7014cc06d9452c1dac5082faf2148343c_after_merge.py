    def edit_config(self, command):
        for cmd in chain(['configure terminal'], to_list(command), ['end']):
            if isinstance(cmd, dict):
                command = cmd['command']
                prompt = cmd['prompt']
                answer = cmd['answer']
                newline = cmd.get('newline', True)
            else:
                command = cmd
                prompt = None
                answer = None
                newline = True

            self.send_command(command, prompt, answer, False, newline)