    def build_entity(self):
        sched_policy = self._get_sched_policy()
        return otypes.Cluster(
            name=self.param('name'),
            comment=self.param('comment'),
            description=self.param('description'),
            ballooning_enabled=self.param('ballooning'),
            gluster_service=self.param('gluster'),
            virt_service=self.param('virt'),
            threads_as_cores=self.param('threads_as_cores'),
            ha_reservation=self.param('ha_reservation'),
            trusted_service=self.param('trusted_service'),
            optional_reason=self.param('vm_reason'),
            maintenance_reason_required=self.param('host_reason'),
            scheduling_policy=otypes.SchedulingPolicy(
                id=sched_policy.id,
            ) if sched_policy else None,
            serial_number=otypes.SerialNumber(
                policy=otypes.SerialNumberPolicy(self.param('serial_policy')),
                value=self.param('serial_policy_value'),
            ) if (
                self.param('serial_policy') is not None or
                self.param('serial_policy_value') is not None
            ) else None,
            migration=otypes.MigrationOptions(
                auto_converge=otypes.InheritableBoolean(
                    self.param('migration_auto_converge'),
                ) if self.param('migration_auto_converge') else None,
                bandwidth=otypes.MigrationBandwidth(
                    assignment_method=otypes.MigrationBandwidthAssignmentMethod(
                        self.param('migration_bandwidth'),
                    ) if self.param('migration_bandwidth') else None,
                    custom_value=self.param('migration_bandwidth_limit'),
                ) if (
                    self.param('migration_bandwidth') or
                    self.param('migration_bandwidth_limit')
                ) else None,
                compressed=otypes.InheritableBoolean(
                    self.param('migration_compressed'),
                ) if self.param('migration_compressed') else None,
                policy=otypes.MigrationPolicy(
                    id=self._get_policy_id()
                ) if self.param('migration_policy') else None,
            ) if (
                self.param('migration_bandwidth') is not None or
                self.param('migration_bandwidth_limit') is not None or
                self.param('migration_auto_converge') is not None or
                self.param('migration_compressed') is not None or
                self.param('migration_policy') is not None
            ) else None,
            error_handling=otypes.ErrorHandling(
                on_error=otypes.MigrateOnError(
                    self.param('resilience_policy')
                ),
            ) if self.param('resilience_policy') else None,
            fencing_policy=otypes.FencingPolicy(
                enabled=self.param('fence_enabled'),
                skip_if_connectivity_broken=otypes.SkipIfConnectivityBroken(
                    enabled=self.param('fence_skip_if_connectivity_broken'),
                    threshold=self.param('fence_connectivity_threshold'),
                ) if (
                    self.param('fence_skip_if_connectivity_broken') is not None or
                    self.param('fence_connectivity_threshold') is not None
                ) else None,
                skip_if_sd_active=otypes.SkipIfSdActive(
                    enabled=self.param('fence_skip_if_sd_active'),
                ) if self.param('fence_skip_if_sd_active') is not None else None,
            ) if (
                self.param('fence_enabled') is not None or
                self.param('fence_skip_if_sd_active') is not None or
                self.param('fence_skip_if_connectivity_broken') is not None or
                self.param('fence_connectivity_threshold') is not None
            ) else None,
            display=otypes.Display(
                proxy=self.param('spice_proxy'),
            ) if self.param('spice_proxy') else None,
            required_rng_sources=[
                otypes.RngSource(rng) for rng in self.param('rng_sources')
            ] if self.param('rng_sources') else None,
            memory_policy=otypes.MemoryPolicy(
                over_commit=otypes.MemoryOverCommit(
                    percent=self._get_memory_policy(),
                ),
            ) if self.param('memory_policy') else None,
            ksm=otypes.Ksm(
                enabled=self.param('ksm'),
                merge_across_nodes=not self.param('ksm_numa'),
            ) if (
                self.param('ksm_numa') is not None or
                self.param('ksm') is not None
            ) else None,
            data_center=otypes.DataCenter(
                name=self.param('data_center'),
            ) if self.param('data_center') else None,
            management_network=otypes.Network(
                name=self.param('network'),
            ) if self.param('network') else None,
            cpu=otypes.Cpu(
                architecture=otypes.Architecture(
                    self.param('cpu_arch')
                ) if self.param('cpu_arch') else None,
                type=self.param('cpu_type'),
            ) if (
                self.param('cpu_arch') or self.param('cpu_type')
            ) else None,
            version=otypes.Version(
                major=self.__get_major(self.param('compatibility_version')),
                minor=self.__get_minor(self.param('compatibility_version')),
            ) if self.param('compatibility_version') else None,
            switch_type=otypes.SwitchType(
                self.param('switch_type')
            ) if self.param('switch_type') else None,
            mac_pool=otypes.MacPool(
                id=get_id_by_name(self._connection.system_service().mac_pools_service(), self.param('mac_pool'))
            ) if self.param('mac_pool') else None,
            external_network_providers=self._get_external_network_providers_entity(),
        )