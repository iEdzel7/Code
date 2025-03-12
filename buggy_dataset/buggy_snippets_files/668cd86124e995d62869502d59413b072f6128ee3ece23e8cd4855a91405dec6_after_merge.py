    def reset(self):
        # If we have a persistent ssh connection (ControlPersist), we can ask it to stop listening.
        cmd = self._build_command(self._play_context.ssh_executable, '-O', 'stop', self.host)
        controlpersist, controlpath = self._persistence_controls(cmd)
        if controlpersist:
            display.vvv(u'sending stop: %s' % cmd)
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = p.communicate()
            status_code = p.wait()
            if status_code != 0:
                raise AnsibleError("Cannot reset connection:\n%s" % stderr)

        self.close()