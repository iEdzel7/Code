    def _csrf_enabled(self):
        csrf_data = self._get_json_data(
            "%s/%s" % (self.url, "api/json"), 'CSRF')

        try:
            return csrf_data["useCrumbs"]
        except:
            self.module.fail_json(
                msg="Required fields not found in the Crum response.",
                details=csrf_data)