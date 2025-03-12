    def read_settings(self):
        ''' Reads the settings from the vmware_inventory.ini file '''

        scriptbasename = __file__
        scriptbasename = os.path.basename(scriptbasename)
        scriptbasename = scriptbasename.replace('.py', '')

        defaults = {'vmware': {
            'server': '',
            'port': 443,
            'username': '',
            'password': '',
            'validate_certs': True,
            'ini_path': os.path.join(os.path.dirname(__file__), '%s.ini' % scriptbasename),
            'cache_name': 'ansible-vmware',
            'cache_path': '~/.ansible/tmp',
            'cache_max_age': 3600,
            'max_object_level': 1,
            'skip_keys': 'declaredalarmstate,'
                         'disabledmethod,'
                         'dynamicproperty,'
                         'dynamictype,'
                         'environmentbrowser,'
                         'managedby,'
                         'parent,'
                         'childtype,'
                         'resourceconfig',
            'alias_pattern': '{{ config.name + "_" + config.uuid }}',
            'host_pattern': '{{ guest.ipaddress }}',
            'host_filters': '{{ runtime.powerstate == "poweredOn" }}',
            'groupby_patterns': '{{ guest.guestid }},{{ "templates" if config.template else "guests"}}',
            'lower_var_keys': True,
            'custom_field_group_prefix': 'vmware_tag_',
            'groupby_custom_field_excludes': '',
            'groupby_custom_field': False}
        }

        if PY3:
            config = configparser.ConfigParser()
        else:
            config = configparser.SafeConfigParser()

        # where is the config?
        vmware_ini_path = os.environ.get('VMWARE_INI_PATH', defaults['vmware']['ini_path'])
        vmware_ini_path = os.path.expanduser(os.path.expandvars(vmware_ini_path))
        config.read(vmware_ini_path)

        if 'vmware' not in config.sections():
            config.add_section('vmware')

        # apply defaults
        for k, v in defaults['vmware'].items():
            if not config.has_option('vmware', k):
                config.set('vmware', k, str(v))

        # where is the cache?
        self.cache_dir = os.path.expanduser(config.get('vmware', 'cache_path'))
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # set the cache filename and max age
        cache_name = config.get('vmware', 'cache_name')
        self.cache_path_cache = self.cache_dir + "/%s.cache" % cache_name
        self.debugl('cache path is %s' % self.cache_path_cache)
        self.cache_max_age = int(config.getint('vmware', 'cache_max_age'))

        # mark the connection info
        self.server = os.environ.get('VMWARE_SERVER', config.get('vmware', 'server'))
        self.debugl('server is %s' % self.server)
        self.port = int(os.environ.get('VMWARE_PORT', config.get('vmware', 'port')))
        self.username = os.environ.get('VMWARE_USERNAME', config.get('vmware', 'username'))
        self.debugl('username is %s' % self.username)
        self.password = os.environ.get('VMWARE_PASSWORD', config.get('vmware', 'password', raw=True))
        self.validate_certs = os.environ.get('VMWARE_VALIDATE_CERTS', config.get('vmware', 'validate_certs'))
        if self.validate_certs in ['no', 'false', 'False', False]:
            self.validate_certs = False

        self.debugl('cert validation is %s' % self.validate_certs)

        # behavior control
        self.maxlevel = int(config.get('vmware', 'max_object_level'))
        self.debugl('max object level is %s' % self.maxlevel)
        self.lowerkeys = config.get('vmware', 'lower_var_keys')
        if type(self.lowerkeys) != bool:
            if str(self.lowerkeys).lower() in ['yes', 'true', '1']:
                self.lowerkeys = True
            else:
                self.lowerkeys = False
        self.debugl('lower keys is %s' % self.lowerkeys)
        self.skip_keys = list(config.get('vmware', 'skip_keys').split(','))
        self.debugl('skip keys is %s' % self.skip_keys)
        temp_host_filters = list(config.get('vmware', 'host_filters').split('}},'))
        for host_filter in temp_host_filters:
            host_filter = host_filter.rstrip()
            if host_filter != "":
                if not host_filter.endswith("}}"):
                    host_filter += "}}"
                self.host_filters.append(host_filter)
        self.debugl('host filters are %s' % self.host_filters)

        temp_groupby_patterns = list(config.get('vmware', 'groupby_patterns').split('}},'))
        for groupby_pattern in temp_groupby_patterns:
            groupby_pattern = groupby_pattern.rstrip()
            if groupby_pattern != "":
                if not groupby_pattern.endswith("}}"):
                    groupby_pattern += "}}"
                self.groupby_patterns.append(groupby_pattern)
        self.debugl('groupby patterns are %s' % self.groupby_patterns)
        temp_groupby_custom_field_excludes = config.get('vmware', 'groupby_custom_field_excludes')
        self.groupby_custom_field_excludes = [x.strip('"') for x in [y.strip("'") for y in temp_groupby_custom_field_excludes.split(",")]]
        self.debugl('groupby exclude strings are %s' % self.groupby_custom_field_excludes)

        # Special feature to disable the brute force serialization of the
        # virtual machine objects. The key name for these properties does not
        # matter because the values are just items for a larger list.
        if config.has_section('properties'):
            self.guest_props = []
            for prop in config.items('properties'):
                self.guest_props.append(prop[1])

        # save the config
        self.config = config