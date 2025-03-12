    def container_start(self, container_id):
        self.log("start container %s" % (container_id))
        self.results['actions'].append(dict(started=container_id))
        self.results['changed'] = True
        if not self.check_mode:
            try:
                self.client.start(container=container_id)
            except Exception as exc:
                self.fail("Error starting container %s: %s" % (container_id, str(exc)))

            if not self.parameters.detach:
                if HAS_DOCKER_PY_3:
                    status = self.client.wait(container_id)['StatusCode']
                else:
                    status = self.client.wait(container_id)
                if self.parameters.auto_remove:
                    output = "Cannot retrieve result as auto_remove is enabled"
                    if self.parameters.output_logs:
                        self.client.module.warn('Cannot output_logs if auto_remove is enabled!')
                else:
                    config = self.client.inspect_container(container_id)
                    logging_driver = config['HostConfig']['LogConfig']['Type']

                    if logging_driver == 'json-file' or logging_driver == 'journald':
                        output = self.client.logs(container_id, stdout=True, stderr=True, stream=False, timestamps=False)
                        if self.parameters.output_logs:
                            self._output_logs(msg=output)
                    else:
                        output = "Result logged using `%s` driver" % logging_driver

                if status != 0:
                    self.fail(output, status=status)
                if self.parameters.cleanup:
                    self.container_remove(container_id, force=True, ignore_failure=self.parameters.auto_remove)
                insp = self._get_container(container_id)
                if insp.raw:
                    insp.raw['Output'] = output
                else:
                    insp.raw = dict(Output=output)
                return insp
        return self._get_container(container_id)