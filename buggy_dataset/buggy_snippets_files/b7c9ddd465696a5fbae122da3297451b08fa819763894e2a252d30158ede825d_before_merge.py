    def _winrm_exec(self, command, args=(), from_exec=False, stdin_iterator=None):
        if not self.protocol:
            self.protocol = self._winrm_connect()
            self._connected = True
        if from_exec:
            display.vvvvv("WINRM EXEC %r %r" % (command, args), host=self._winrm_host)
        else:
            display.vvvvvv("WINRM EXEC %r %r" % (command, args), host=self._winrm_host)
        command_id = None
        try:
            stdin_push_failed = False
            command_id = self.protocol.run_command(self.shell_id, to_bytes(command), map(to_bytes, args), console_mode_stdin=(stdin_iterator is None))

            # TODO: try/except around this, so we can get/return the command result on a broken pipe or other failure (probably more useful than the 500 that
            # comes from this)
            try:
                if stdin_iterator:
                    for (data, is_last) in stdin_iterator:
                        self._winrm_send_input(self.protocol, self.shell_id, command_id, data, eof=is_last)

            except Exception as ex:
                from traceback import format_exc
                display.warning("FATAL ERROR DURING FILE TRANSFER: %s" % format_exc())
                stdin_push_failed = True

            if stdin_push_failed:
                raise AnsibleError('winrm send_input failed')

            # NB: this can hang if the receiver is still running (eg, network failed a Send request but the server's still happy).
            # FUTURE: Consider adding pywinrm status check/abort operations to see if the target is still running after a failure.
            resptuple = self.protocol.get_command_output(self.shell_id, command_id)
            # ensure stdout/stderr are text for py3
            # FUTURE: this should probably be done internally by pywinrm
            response = Response(tuple(to_text(v) if isinstance(v, binary_type) else v for v in resptuple))

            # TODO: check result from response and set stdin_push_failed if we have nonzero
            if from_exec:
                display.vvvvv('WINRM RESULT %r' % to_text(response), host=self._winrm_host)
            else:
                display.vvvvvv('WINRM RESULT %r' % to_text(response), host=self._winrm_host)

            display.vvvvvv('WINRM STDOUT %s' % to_text(response.std_out), host=self._winrm_host)
            display.vvvvvv('WINRM STDERR %s' % to_text(response.std_err), host=self._winrm_host)

            if stdin_push_failed:
                raise AnsibleError('winrm send_input failed; \nstdout: %s\nstderr %s' % (response.std_out, response.std_err))

            return response
        finally:
            if command_id:
                self.protocol.cleanup_command(self.shell_id, command_id)