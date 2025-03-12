    def init_swarm(self):
        if self.__isSwarmManager():
            self.__update_swarm()
            return

        if not self.check_mode:
            try:
                self.client.init_swarm(
                    advertise_addr=self.parameters.advertise_addr, listen_addr=self.parameters.listen_addr,
                    force_new_cluster=self.parameters.force_new_cluster, swarm_spec=self.parameters.spec)
            except APIError as exc:
                self.client.fail("Can not create a new Swarm Cluster: %s" % to_native(exc))

        self.__isSwarmManager()
        self.results['actions'].append("New Swarm cluster created: %s" % (self.swarm_info.get('ID')))
        self.results['changed'] = True
        self.results['swarm_facts'] = {u'JoinTokens': self.swarm_info.get('JoinTokens')}