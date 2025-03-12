    def _update_check(self, entity):
        return (
            equal(self._module.params.get('description'), entity.description) and
            equal(self.param('quota_id'), getattr(entity.quota, 'id')) and
            equal(convert_to_bytes(self._module.params.get('size')), entity.provisioned_size) and
            equal(self._module.params.get('shareable'), entity.shareable)
        )