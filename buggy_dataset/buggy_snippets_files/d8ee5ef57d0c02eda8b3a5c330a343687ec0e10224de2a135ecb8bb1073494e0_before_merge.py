    def _credentials(self):
        cred_type = self.module.params['auth_kind']
        if cred_type == 'application':
            credentials, project_id = google.auth.default(scopes=self.module.params['scopes'])
            return credentials
        elif cred_type == 'serviceaccount' and self.module.params.get('service_account_file'):
            path = os.path.realpath(os.path.expanduser(self.module.params['service_account_file']))
            return service_account.Credentials.from_service_account_file(path).with_scopes(self.module.params['scopes'])
        elif cred_type == 'serviceaccount' and self.module.params.get('service_account_contents'):
            cred = json.loads(self.module.params.get('service_account_contents'))
            return service_account.Credentials.from_service_account_info(cred).with_scopes(self.module.params['scopes'])
        elif cred_type == 'machineaccount':
            return google.auth.compute_engine.Credentials(
                self.module.params['service_account_email'])
        else:
            self.module.fail_json(msg="Credential type '%s' not implemented" % cred_type)