    def deploy_template(self):
        """
        Deploy the targeted template and parameters
        :param module: Ansible module containing the validated configuration for the deployment template
        :param client: resource management client for azure
        :param conn_info: connection info needed
        :return:
        """

        deploy_parameter = self.rm_models.DeploymentProperties(self.deployment_mode)
        if not self.parameters_link:
            deploy_parameter.parameters = self.parameters
        else:
            deploy_parameter.parameters_link = self.rm_models.ParametersLink(
                uri=self.parameters_link
            )
        if not self.template_link:
            deploy_parameter.template = self.template
        else:
            deploy_parameter.template_link = self.rm_models.TemplateLink(
                uri=self.template_link
            )

        if self.append_tags and self.tags:
            try:
                rg = self.get_resource_group(self.resource_group_name)
                if rg.tags:
                    self.tags = dict(self.tags, **rg.tags)
            except CloudError:
                # resource group does not exist
                pass

        params = self.rm_models.ResourceGroup(location=self.location, tags=self.tags)

        try:
            self.rm_client.resource_groups.create_or_update(self.resource_group_name, params)
        except CloudError as exc:
            self.fail("Resource group create_or_update failed with status code: %s and message: %s" %
                      (exc.status_code, exc.message))
        try:
            result = self.rm_client.deployments.create_or_update(self.resource_group_name,
                                                                 self.deployment_name,
                                                                 deploy_parameter)

            deployment_result = None
            if self.wait_for_deployment_completion:
                deployment_result = self.get_poller_result(result)
                while deployment_result.properties is None or deployment_result.properties.provisioning_state not in ['Canceled', 'Failed', 'Deleted',
                                                                                                                      'Succeeded']:
                    time.sleep(self.wait_for_deployment_polling_period)
                    deployment_result = self.rm_client.deployments.get(self.resource_group_name, self.deployment_name)
        except CloudError as exc:
            failed_deployment_operations = self._get_failed_deployment_operations(self.deployment_name)
            self.log("Deployment failed %s: %s" % (exc.status_code, exc.message))
            self.fail("Deployment failed with status code: %s and message: %s" % (exc.status_code, exc.message),
                      failed_deployment_operations=failed_deployment_operations)

        if self.wait_for_deployment_completion and deployment_result.properties.provisioning_state != 'Succeeded':
            self.log("provisioning state: %s" % deployment_result.properties.provisioning_state)
            failed_deployment_operations = self._get_failed_deployment_operations(self.deployment_name)
            self.fail('Deployment failed. Deployment id: %s' % deployment_result.id,
                      failed_deployment_operations=failed_deployment_operations)

        return deployment_result