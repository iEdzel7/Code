    def deploy_vm(self):
        # https://github.com/vmware/pyvmomi-community-samples/blob/master/samples/clone_vm.py
        # https://www.vmware.com/support/developer/vc-sdk/visdk25pubs/ReferenceGuide/vim.vm.CloneSpec.html
        # https://www.vmware.com/support/developer/vc-sdk/visdk25pubs/ReferenceGuide/vim.vm.ConfigSpec.html
        # https://www.vmware.com/support/developer/vc-sdk/visdk41pubs/ApiReference/vim.vm.RelocateSpec.html

        # FIXME:
        #   - multiple templates by the same name
        #   - static IPs

        # datacenters = get_all_objs(self.content, [vim.Datacenter])
        datacenter = self.cache.find_obj(self.content, [vim.Datacenter], self.params['datacenter'])
        if datacenter is None:
            self.module.fail_json(msg='No datacenter named %(datacenter)s was found' % self.params)

        # Prepend / if it was missing from the folder path, also strip trailing slashes
        if not self.params['folder'].startswith('/'):
            self.params['folder'] = '/%(folder)s' % self.params
        self.params['folder'] = self.params['folder'].rstrip('/')

        dcpath = compile_folder_path_for_object(datacenter)

        # Check for full path first in case it was already supplied
        if (self.params['folder'].startswith(dcpath + self.params['datacenter'] + '/vm')):
            fullpath = self.params['folder']
        elif (self.params['folder'].startswith('/vm/') or self.params['folder'] == '/vm'):
            fullpath = "%s%s%s" % (dcpath, self.params['datacenter'], self.params['folder'])
        elif (self.params['folder'].startswith('/')):
            fullpath = "%s%s/vm%s" % (dcpath, self.params['datacenter'], self.params['folder'])
        else:
            fullpath = "%s%s/vm/%s" % (dcpath, self.params['datacenter'], self.params['folder'])

        f_obj = self.content.searchIndex.FindByInventoryPath(fullpath)

        # abort if no strategy was successful
        if f_obj is None:
            self.module.fail_json(msg='No folder matched the path: %(folder)s' % self.params)
        destfolder = f_obj

        if self.params['template']:
            # FIXME: need to search for this in the same way as guests to ensure accuracy
            vm_obj = find_obj(self.content, [vim.VirtualMachine], self.params['template'])
            if vm_obj is None:
                self.module.fail_json(msg="Could not find a template named %(template)s" % self.params)
        else:
            vm_obj = None

        # need a resource pool if cloning from template
        if self.params['resource_pool'] or self.params['template']:
            resource_pool = self.get_resource_pool()

        # set the destination datastore for VM & disks
        (datastore, datastore_name) = self.select_datastore(vm_obj)

        self.configspec = vim.vm.ConfigSpec(cpuHotAddEnabled=True, memoryHotAddEnabled=True)
        self.configspec.deviceChange = []
        self.configure_guestid(vm_obj=vm_obj, vm_creation=True)
        self.configure_cpu_and_memory(vm_obj=vm_obj, vm_creation=True)
        self.configure_disks(vm_obj=vm_obj)
        self.configure_network(vm_obj=vm_obj)

        # Find if we need network customizations (find keys in dictionary that requires customizations)
        network_changes = False
        for nw in self.params['networks']:
            for key in nw:
                # We don't need customizations for these keys
                if key not in ('device_type', 'mac', 'name', 'vlan'):
                    network_changes = True
                    break

        if len(self.params['customization']) > 0 or network_changes is True:
            self.customize_vm(vm_obj=vm_obj)

        clonespec = None
        clone_method = None
        try:
            if self.params['template']:
                # create the relocation spec
                relospec = vim.vm.RelocateSpec()

                # Only select specific host when ESXi hostname is provided
                if self.params['esxi_hostname']:
                    relospec.host = self.select_host()
                relospec.datastore = datastore

                # https://www.vmware.com/support/developer/vc-sdk/visdk41pubs/ApiReference/vim.vm.RelocateSpec.html
                # > pool: For a clone operation from a template to a virtual machine, this argument is required.
                relospec.pool = resource_pool

                if self.params['snapshot_src'] is not None and self.params['linked_clone']:
                    relospec.diskMoveType = vim.vm.RelocateSpec.DiskMoveOptions.createNewChildDiskBacking

                clonespec = vim.vm.CloneSpec(template=self.params['is_template'], location=relospec)
                if self.customspec:
                    clonespec.customization = self.customspec

                if self.params['snapshot_src'] is not None:
                    snapshot = self.get_snapshots_by_name_recursively(snapshots=vm_obj.snapshot.rootSnapshotList, snapname=self.params['snapshot_src'])
                    if len(snapshot) != 1:
                        self.module.fail_json(msg='virtual machine "%(template)s" does not contain snapshot named "%(snapshot_src)s"' % self.params)

                    clonespec.snapshot = snapshot[0].snapshot

                clonespec.config = self.configspec
                clone_method = 'Clone'
                task = vm_obj.Clone(folder=destfolder, name=self.params['name'], spec=clonespec)
                self.change_detected = True
            else:
                # ConfigSpec require name for VM creation
                self.configspec.name = self.params['name']
                self.configspec.files = vim.vm.FileInfo(logDirectory=None,
                                                        snapshotDirectory=None,
                                                        suspendDirectory=None,
                                                        vmPathName="[" + datastore_name + "] " + self.params["name"])

                clone_method = 'CreateVM_Task'
                resource_pool = self.get_resource_pool()
                task = destfolder.CreateVM_Task(config=self.configspec, pool=resource_pool)
                self.change_detected = True
            self.wait_for_task(task)
        except TypeError as e:
            self.module.fail_json(msg="TypeError was returned, please ensure to give correct inputs. %s" % to_text(e))

        if task.info.state == 'error':
            # https://kb.vmware.com/selfservice/microsites/search.do?language=en_US&cmd=displayKC&externalId=2021361
            # https://kb.vmware.com/selfservice/microsites/search.do?language=en_US&cmd=displayKC&externalId=2173

            # provide these to the user for debugging
            clonespec_json = serialize_spec(clonespec)
            configspec_json = serialize_spec(self.configspec)
            kwargs = {
                'changed': self.change_detected,
                'failed': True,
                'msg': task.info.error.msg,
                'clonespec': clonespec_json,
                'configspec': configspec_json,
                'clone_method': clone_method
            }

            return kwargs
        else:
            # set annotation
            vm = task.info.result
            if self.params['annotation']:
                annotation_spec = vim.vm.ConfigSpec()
                annotation_spec.annotation = str(self.params['annotation'])
                task = vm.ReconfigVM_Task(annotation_spec)
                self.wait_for_task(task)

            self.customize_customvalues(vm_obj=vm)

            if self.params['wait_for_ip_address'] or self.params['state'] in ['poweredon', 'restarted']:
                self.set_powerstate(vm, 'poweredon', force=False)

                if self.params['wait_for_ip_address']:
                    self.wait_for_vm_ip(vm)

            vm_facts = self.gather_facts(vm)
            return {'changed': self.change_detected, 'failed': False, 'instance': vm_facts}