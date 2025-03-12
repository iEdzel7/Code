    def process_template(self, cfg, minimize=None):
        """
        Process the Policy Sentry template as a dict. This auto-detects whether or not the file is in CRUD mode or
        Actions mode.

        Arguments:
            cfg: The loaded YAML as a dict. Must follow Policy Sentry dictated format.
            minimize: Minimize the resulting statement with *safe* usage of wildcards to reduce policy length. Set this to the character length you want - for example, 0, or 4. Defaults to none.
        Returns:
            Dictionary: The rendered IAM JSON Policy
        """
        if cfg.get("mode") == "crud":
            logger.debug("CRUD mode selected")
            check_crud_schema(cfg)
            # EXCLUDE ACTIONS
            if cfg.get("exclude-actions"):
                if cfg.get("exclude-actions")[0] != "":
                    self.add_exclude_actions(cfg["exclude-actions"])
            # WILDCARD ONLY SECTION
            if cfg.get("wildcard-only"):
                if cfg.get("wildcard-only").get("single-actions"):
                    if cfg["wildcard-only"]["single-actions"][0] != "":
                        provided_wildcard_actions = cfg["wildcard-only"]["single-actions"]
                        logger.debug(f"Requested wildcard-only actions: {str(provided_wildcard_actions)}")
                        self.wildcard_only_single_actions = provided_wildcard_actions
                if cfg.get("wildcard-only").get("service-read"):
                    if cfg["wildcard-only"]["service-read"][0] != "":
                        service_read = cfg["wildcard-only"]["service-read"]
                        logger.debug(f"Requested wildcard-only actions: {str(service_read)}")
                        self.wildcard_only_service_read = service_read
                if cfg.get("wildcard-only").get("service-write"):
                    if cfg["wildcard-only"]["service-write"][0] != "":
                        service_write = cfg["wildcard-only"]["service-write"]
                        logger.debug(f"Requested wildcard-only actions: {str(service_write)}")
                        self.wildcard_only_service_write = service_write
                if cfg.get("wildcard-only").get("service-list"):
                    if cfg["wildcard-only"]["service-list"][0] != "":
                        service_list = cfg["wildcard-only"]["service-list"]
                        logger.debug(f"Requested wildcard-only actions: {str(service_list)}")
                        self.wildcard_only_service_list = service_list
                if cfg.get("wildcard-only").get("service-tagging"):
                    if cfg["wildcard-only"]["service-tagging"][0] != "":
                        service_tagging = cfg["wildcard-only"]["service-tagging"]
                        logger.debug(f"Requested wildcard-only actions: {str(service_tagging)}")
                        self.wildcard_only_service_tagging = service_tagging
                if cfg.get("wildcard-only").get("service-permissions-management"):
                    if cfg["wildcard-only"]["service-permissions-management"][0] != "":
                        service_permissions_management = cfg["wildcard-only"]["service-permissions-management"]
                        logger.debug(f"Requested wildcard-only actions: {str(service_permissions_management)}")
                        self.wildcard_only_service_permissions_management = service_permissions_management

            # Process the wildcard-only section
            self.process_wildcard_only_actions()

            # Standard access levels
            if cfg.get("read"):
                if cfg["read"][0] != "":
                    logger.debug(f"Requested access to arns: {str(cfg['read'])}")
                    self.add_by_arn_and_access_level(cfg["read"], "Read")
            if cfg.get("write"):
                if cfg["write"][0] != "":
                    logger.debug(f"Requested access to arns: {str(cfg['write'])}")
                    self.add_by_arn_and_access_level(cfg["write"], "Write")
            if cfg.get("list"):
                if cfg["list"][0] != "":
                    logger.debug(f"Requested access to arns: {str(cfg['list'])}")
                    self.add_by_arn_and_access_level(cfg["list"], "List")
            if cfg.get("tagging"):
                if cfg["tagging"][0] != "":
                    logger.debug(f"Requested access to arns: {str(cfg['tagging'])}")
                    self.add_by_arn_and_access_level(cfg["tagging"], "Tagging")
            if cfg.get("permissions-management"):
                if cfg["permissions-management"][0] != "":
                    logger.debug(f"Requested access to arns: {str(cfg['permissions-management'])}")
                    self.add_by_arn_and_access_level(cfg["permissions-management"], "Permissions management")

            # SKIP RESOURCE CONSTRAINTS
            if cfg.get("skip-resource-constraints"):
                if cfg["skip-resource-constraints"][0] != "":
                    logger.debug(
                        f"Requested override: the actions {str(cfg['skip-resource-constraints'])} will "
                        f"skip resource constraints."
                    )
                    self.add_skip_resource_constraints(cfg["skip-resource-constraints"])
                    for skip_resource_constraints_action in self.skip_resource_constraints:
                        self.add_action_without_resource_constraint(
                            skip_resource_constraints_action, "SkipResourceConstraints"
                        )
        elif cfg.get("mode") == "actions":
            check_actions_schema(cfg)
            if "actions" in cfg.keys():
                if cfg["actions"] is not None and cfg["actions"][0] != "":
                    self.add_by_list_of_actions(cfg["actions"])

        rendered_policy = self.get_rendered_policy(minimize)
        return rendered_policy