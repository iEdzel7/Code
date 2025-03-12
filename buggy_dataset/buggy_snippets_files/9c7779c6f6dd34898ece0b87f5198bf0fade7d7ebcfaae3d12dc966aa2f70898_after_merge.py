    def get_api_client(self, **auth_params):
        auth_args = AUTH_ARG_SPEC.keys()

        auth_params = auth_params or getattr(self, 'params', {})
        auth = copy.deepcopy(auth_params)

        # If authorization variables aren't defined, look for them in environment variables
        for key, value in iteritems(auth_params):
            if key in auth_args and value is None:
                env_value = os.getenv('K8S_AUTH_{0}'.format(key.upper()), None)
                if env_value is not None:
                    auth[key] = env_value

        def auth_set(*names):
            return all([auth.get(name) for name in names])

        if auth_set('username', 'password', 'host') or auth_set('api_key', 'host'):
            # We have enough in the parameters to authenticate, no need to load incluster or kubeconfig
            pass
        elif auth_set('kubeconfig', 'context'):
            kubernetes.config.load_kube_config(auth.get('kubeconfig'), auth.get('context'))
        else:
            # First try to do incluster config, then kubeconfig
            try:
                kubernetes.config.load_incluster_config()
            except kubernetes.config.ConfigException:
                kubernetes.config.load_kube_config(auth.get('kubeconfig'), auth.get('context'))

        # Override any values in the default configuration with Ansible parameters
        configuration = kubernetes.client.Configuration()
        for key, value in iteritems(auth):
            if key in auth_args and value is not None:
                if key == 'api_key':
                    setattr(configuration, key, {'authorization': "Bearer {0}".format(value)})
                else:
                    setattr(configuration, key, value)

        kubernetes.client.Configuration.set_default(configuration)
        return DynamicClient(kubernetes.client.ApiClient(configuration))