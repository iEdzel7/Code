    def set_host_overrides(self, host, hostvars=None):
        '''
        Override WinRM-specific options from host variables.
        '''
        if not HAS_WINRM:
            return

        self._winrm_host = self._play_context.remote_addr
        self._winrm_port = int(self._play_context.port or 5986)
        self._winrm_scheme = hostvars.get('ansible_winrm_scheme', 'http' if self._winrm_port == 5985 else 'https')
        self._winrm_path = hostvars.get('ansible_winrm_path', '/wsman')
        self._winrm_user = self._play_context.remote_user
        self._winrm_pass = self._play_context.password
        self._become_method = self._play_context.become_method
        self._become_user = self._play_context.become_user
        self._become_pass = self._play_context.become_pass

        self._kinit_cmd = hostvars.get('ansible_winrm_kinit_cmd', 'kinit')

        if hasattr(winrm, 'FEATURE_SUPPORTED_AUTHTYPES'):
            self._winrm_supported_authtypes = set(winrm.FEATURE_SUPPORTED_AUTHTYPES)
        else:
            # for legacy versions of pywinrm, use the values we know are supported
            self._winrm_supported_authtypes = set(['plaintext', 'ssl', 'kerberos'])

        # TODO: figure out what we want to do with auto-transport selection in the face of NTLM/Kerb/CredSSP/Cert/Basic
        transport_selector = 'ssl' if self._winrm_scheme == 'https' else 'plaintext'

        if HAVE_KERBEROS and ((self._winrm_user and '@' in self._winrm_user)):
            self._winrm_transport = 'kerberos,%s' % transport_selector
        else:
            self._winrm_transport = transport_selector
        self._winrm_transport = hostvars.get('ansible_winrm_transport', self._winrm_transport)
        if isinstance(self._winrm_transport, string_types):
            self._winrm_transport = [x.strip() for x in self._winrm_transport.split(',') if x.strip()]

        unsupported_transports = set(self._winrm_transport).difference(self._winrm_supported_authtypes)

        if unsupported_transports:
            raise AnsibleError('The installed version of WinRM does not support transport(s) %s' % list(unsupported_transports))

        # if kerberos is among our transports and there's a password specified, we're managing the tickets
        kinit_mode = to_text(hostvars.get('ansible_winrm_kinit_mode', '')).strip()
        if kinit_mode == "":
            # HACK: ideally, remove multi-transport stuff
            self._kerb_managed = "kerberos" in self._winrm_transport and self._winrm_pass
        elif kinit_mode == "managed":
            self._kerb_managed = True
        elif kinit_mode == "manual":
            self._kerb_managed = False
        else:
            raise AnsibleError('Unknown ansible_winrm_kinit_mode value: "%s" (must be "managed" or "manual")' % kinit_mode)

        # arg names we're going passing directly
        internal_kwarg_mask = set(['self', 'endpoint', 'transport', 'username', 'password', 'scheme', 'path', 'kinit_mode', 'kinit_cmd'])

        self._winrm_kwargs = dict(username=self._winrm_user, password=self._winrm_pass)
        argspec = inspect.getargspec(Protocol.__init__)
        supported_winrm_args = set(argspec.args)
        supported_winrm_args.update(internal_kwarg_mask)
        passed_winrm_args = set([v.replace('ansible_winrm_', '') for v in hostvars if v.startswith('ansible_winrm_')])
        unsupported_args = passed_winrm_args.difference(supported_winrm_args)

        # warn for kwargs unsupported by the installed version of pywinrm
        for arg in unsupported_args:
            display.warning("ansible_winrm_{0} unsupported by pywinrm (is an up-to-date version of pywinrm installed?)".format(arg))

        # pass through matching kwargs, excluding the list we want to treat specially
        for arg in passed_winrm_args.difference(internal_kwarg_mask).intersection(supported_winrm_args):
            self._winrm_kwargs[arg] = hostvars['ansible_winrm_%s' % arg]