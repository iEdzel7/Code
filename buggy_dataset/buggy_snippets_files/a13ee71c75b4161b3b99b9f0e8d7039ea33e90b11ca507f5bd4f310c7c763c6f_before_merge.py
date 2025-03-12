    def _get_instances_from_children(self, child):
        instances = []

        if hasattr(child, 'childEntity'):
            self.debugl("CHILDREN: %s" % str(child.childEntity))
            instances += self._get_instances_from_children(child.childEntity)
        elif hasattr(child, 'vmFolder'):
            self.debugl("FOLDER: %s" % str(child))
            instances += self._get_instances_from_children(child.vmFolder)
        elif hasattr(child, 'index'):
            self.debugl("LIST: %s" % str(child))
            for x in sorted(child):
                self.debugl("LIST_ITEM: %s" % x)
                instances += self._get_instances_from_children(x)
        elif hasattr(child, 'guest'):
            self.debugl("GUEST: %s" % str(child))
            instances.append(child)
        elif hasattr(child, 'vm'):    
            # resource pools
            self.debugl("RESOURCEPOOL: %s" % child.vm)
            if child.vm:
                instances += self._get_instances_from_children(child.vm)
        else:            
            self.debugl("ELSE ...")
            try:
                self.debugl(child.__dict__)
            except Exception as e:
                pass
            self.debugl(child)
        return instances