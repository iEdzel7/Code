    def gather_host_vmnic_facts(self):
        hosts_vmnic_facts = {}
        for host in self.hosts:
            host_vmnic_facts = dict(all=[], available=[], used=[], vswitch=dict(), dvswitch=dict())
            host_nw_system = host.configManager.networkSystem
            if host_nw_system:
                nw_config = host_nw_system.networkConfig
                host_vmnic_facts['all'] = [pnic.device for pnic in nw_config.pnic]

                vswitch_vmnics = []
                proxy_switch_vmnics = []
                if nw_config.vswitch:
                    for vswitch in nw_config.vswitch:
                        host_vmnic_facts['vswitch'][vswitch.name] = []
                        for vnic in vswitch.spec.bridge.nicDevice:
                            vswitch_vmnics.append(vnic)
                            host_vmnic_facts['vswitch'][vswitch.name].append(vnic)

                if nw_config.proxySwitch:
                    for proxy_config in nw_config.proxySwitch:
                        dvs_obj = self.find_dvs_by_uuid(uuid=proxy_config.uuid)
                        if dvs_obj:
                            host_vmnic_facts['dvswitch'][dvs_obj.name] = []
                        for proxy_nic in proxy_config.spec.backing.pnicSpec:
                            proxy_switch_vmnics.append(proxy_nic.pnicDevice)
                            if dvs_obj:
                                host_vmnic_facts['dvswitch'][dvs_obj.name].append(proxy_nic.pnicDevice)

                used_vmics = proxy_switch_vmnics + vswitch_vmnics
                host_vmnic_facts['used'] = used_vmics
                host_vmnic_facts['available'] = [pnic.device for pnic in nw_config.pnic if pnic.device not in used_vmics]

            hosts_vmnic_facts[host.name] = host_vmnic_facts
        return hosts_vmnic_facts