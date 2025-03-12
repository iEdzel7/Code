    def make_become_cmd(self, cmd, executable=None):
        """ helper function to create privilege escalation commands """

        prompt      = None
        success_key = None
        self.prompt = None

        if self.become:

            if not executable:
                executable = self.executable

            becomecmd   = None
            randbits    = ''.join(random.choice(string.ascii_lowercase) for x in range(32))
            success_key = 'BECOME-SUCCESS-%s' % randbits
            success_cmd = pipes.quote('echo %s; %s' % (success_key, cmd))

            if executable:
                command = '%s -c %s' % (executable, success_cmd)
            else:
                command = success_cmd

            # set executable to use for the privilege escalation method, with various overrides
            exe = self.become_exe or \
                  getattr(self, '%s_exe' % self.become_method, None) or \
                  C.DEFAULT_BECOME_EXE or \
                  getattr(C, 'DEFAULT_%s_EXE' % self.become_method.upper(), None) or \
                  self.become_method

            # set flags to use for the privilege escalation method, with various overrides
            flags = self.become_flags or \
                    getattr(self, '%s_flags' % self.become_method, None) or \
                    C.DEFAULT_BECOME_FLAGS or \
                    getattr(C, 'DEFAULT_%s_FLAGS' % self.become_method.upper(), None) or \
                    ''

            if self.become_method == 'sudo':
                # If we have a password, we run sudo with a randomly-generated
                # prompt set using -p. Otherwise we run it with default -n, which makes
                # it fail if it would have prompted for a password.
                # Cannot rely on -n as it can be removed from defaults, which should be
                # done for older versions of sudo that do not support the option.
                #
                # Passing a quoted compound command to sudo (or sudo -s)
                # directly doesn't work, so we shellquote it with pipes.quote()
                # and pass the quoted string to the user's shell.

                # force quick error if password is required but not supplied, should prevent sudo hangs.
                if self.become_pass:
                    prompt = '[sudo via ansible, key=%s] password: ' % randbits
                    becomecmd = '%s %s -p "%s" -u %s %s' % (exe,  flags.replace('-n',''), prompt, self.become_user, command)
                else:
                    becomecmd = '%s %s -u %s %s' % (exe, flags, self.become_user, command)


            elif self.become_method == 'su':

                # passing code ref to examine prompt as simple string comparisson isn't good enough with su
                def detect_su_prompt(b_data):
                    b_SU_PROMPT_LOCALIZATIONS_RE = re.compile(b"|".join([b'(\w+\'s )?' + x + b' ?: ?' for x in b_SU_PROMPT_LOCALIZATIONS]), flags=re.IGNORECASE)
                    return bool(b_SU_PROMPT_LOCALIZATIONS_RE.match(b_data))
                prompt = detect_su_prompt

                becomecmd = '%s %s %s -c %s' % (exe, flags, self.become_user, pipes.quote(command))

            elif self.become_method == 'pbrun':

                prompt='assword:'
                becomecmd = '%s -b %s -u %s %s' % (exe, flags, self.become_user, success_cmd)

            elif self.become_method == 'ksu':
                def detect_ksu_prompt(b_data):
                    return re.match(b"Kerberos password for .*@.*:", b_data)

                prompt = detect_ksu_prompt
                becomecmd = '%s %s %s -e %s' % (exe, self.become_user, flags, command)

            elif self.become_method == 'pfexec':

                # No user as it uses it's own exec_attr to figure it out
                becomecmd = '%s %s "%s"' % (exe, flags, success_cmd)

            elif self.become_method == 'runas':
                raise AnsibleError("'runas' is not yet implemented")
                #FIXME: figure out prompt
                # this is not for use with winrm plugin but if they ever get ssh native on windoez
                becomecmd = '%s %s /user:%s "%s"' % (exe, flags, self.become_user, success_cmd)

            elif self.become_method == 'doas':

                prompt = 'doas (%s@' % self.remote_user
                exe = self.become_exe or 'doas'

                if not self.become_pass:
                    flags += ' -n '

                if self.become_user:
                    flags += ' -u %s ' % self.become_user

                #FIXME: make shell independant
                becomecmd = '%s %s echo %s && %s %s env ANSIBLE=true %s' % (exe, flags, success_key, exe, flags, cmd)

            elif self.become_method == 'dzdo':

                exe = self.become_exe or 'dzdo'
                if self.become_pass:
                    prompt = '[dzdo via ansible, key=%s] password: ' % randbits
                    becomecmd = '%s -p %s -u %s %s' % (exe, pipes.quote(prompt), self.become_user, command)
                else:
                    becomecmd = '%s -u %s %s' % (exe, self.become_user, command)

            else:
                raise AnsibleError("Privilege escalation method not found: %s" % self.become_method)

            if self.become_pass:
                self.prompt = prompt
            self.success_key = success_key
            return becomecmd

        return cmd