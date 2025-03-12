    def get_service_mgr_facts(self):
        #TODO: detect more custom init setups like bootscripts, dmd, s6, Epoch, runit, etc
        # also other OSs other than linux might need to check across several possible candidates

        # try various forms of querying pid 1
        proc_1 = get_file_content('/proc/1/comm')
        if proc_1 is None:
            rc, proc_1, err = self.module.run_command("ps -p 1 -o comm|tail -n 1", use_unsafe_shell=True)
        else:
            proc_1 = os.path.basename(proc_1)

        # The ps command above may return "COMMAND" if the user cannot read /proc, e.g. with grsecurity
        if proc_1 == "COMMAND\n":
            proc_1 = None

        if proc_1 is not None:
            proc_1 = to_native(proc_1)
            proc_1 = proc_1.strip()

        if proc_1 is not None and (proc_1 == 'init' or proc_1.endswith('sh')):
            # many systems return init, so this cannot be trusted, if it ends in 'sh' it probalby is a shell in a container
            proc_1 = None

        # if not init/None it should be an identifiable or custom init, so we are done!
        if proc_1 is not None:
            self.facts['service_mgr'] = proc_1

        # start with the easy ones
        elif  self.facts['distribution'] == 'MacOSX':
            #FIXME: find way to query executable, version matching is not ideal
            if LooseVersion(platform.mac_ver()[0]) >= LooseVersion('10.4'):
                self.facts['service_mgr'] = 'launchd'
            else:
                self.facts['service_mgr'] = 'systemstarter'
        elif 'BSD' in self.facts['system'] or self.facts['system'] in ['Bitrig', 'DragonFly']:
            #FIXME: we might want to break out to individual BSDs
            self.facts['service_mgr'] = 'bsdinit'
        elif self.facts['system'] == 'AIX':
            self.facts['service_mgr'] = 'src'
        elif self.facts['system'] == 'SunOS':
            #FIXME: smf?
            self.facts['service_mgr'] = 'svcs'
        elif self.facts['system'] == 'Linux':
            if self.is_systemd_managed():
                self.facts['service_mgr'] = 'systemd'
            elif self.module.get_bin_path('initctl') and os.path.exists("/etc/init/"):
                self.facts['service_mgr'] = 'upstart'
            elif os.path.realpath('/sbin/rc') == '/sbin/openrc':
                self.facts['service_mgr'] = 'openrc'
            elif os.path.exists('/etc/init.d/'):
                self.facts['service_mgr'] = 'sysvinit'

        if not self.facts.get('service_mgr', False):
            # if we cannot detect, fallback to generic 'service'
            self.facts['service_mgr'] = 'service'