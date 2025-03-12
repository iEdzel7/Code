    def exec_command(self, cmd, in_data=None, sudoable=True):
        ''' run a command on the local host '''

        super(Connection, self).exec_command(cmd, in_data=in_data, sudoable=sudoable)

        display.debug("in local.exec_command()")

        executable = C.DEFAULT_EXECUTABLE.split()[0] if C.DEFAULT_EXECUTABLE else None

        display.vvv(u"EXEC {0}".format(cmd), host=self._play_context.remote_addr)
        display.debug("opening command with Popen()")

        if isinstance(cmd, (text_type, binary_type)):
            cmd = to_bytes(cmd)
        else:
            cmd = map(to_bytes, cmd)

        p = subprocess.Popen(
            cmd,
            shell=isinstance(cmd, (text_type, binary_type)),
            executable=executable, #cwd=...
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        display.debug("done running command with Popen()")

        if self._play_context.prompt and sudoable:
            fcntl.fcntl(p.stdout, fcntl.F_SETFL, fcntl.fcntl(p.stdout, fcntl.F_GETFL) | os.O_NONBLOCK)
            fcntl.fcntl(p.stderr, fcntl.F_SETFL, fcntl.fcntl(p.stderr, fcntl.F_GETFL) | os.O_NONBLOCK)
            become_output = ''
            while not self.check_become_success(become_output) and not self.check_password_prompt(become_output):

                rfd, wfd, efd = select.select([p.stdout, p.stderr], [], [p.stdout, p.stderr], self._play_context.timeout)
                if p.stdout in rfd:
                    chunk = p.stdout.read()
                elif p.stderr in rfd:
                    chunk = p.stderr.read()
                else:
                    stdout, stderr = p.communicate()
                    raise AnsibleError('timeout waiting for privilege escalation password prompt:\n' + become_output)
                if not chunk:
                    stdout, stderr = p.communicate()
                    raise AnsibleError('privilege output closed while waiting for password prompt:\n' + become_output)
                become_output += chunk
            if not self.check_become_success(become_output):
                p.stdin.write(to_bytes(self._play_context.become_pass, errors='surrogate_or_strict') + b'\n')
            fcntl.fcntl(p.stdout, fcntl.F_SETFL, fcntl.fcntl(p.stdout, fcntl.F_GETFL) & ~os.O_NONBLOCK)
            fcntl.fcntl(p.stderr, fcntl.F_SETFL, fcntl.fcntl(p.stderr, fcntl.F_GETFL) & ~os.O_NONBLOCK)

        display.debug("getting output with communicate()")
        stdout, stderr = p.communicate(in_data)
        display.debug("done communicating")

        display.debug("done with local.exec_command()")
        return (p.returncode, stdout, stderr)