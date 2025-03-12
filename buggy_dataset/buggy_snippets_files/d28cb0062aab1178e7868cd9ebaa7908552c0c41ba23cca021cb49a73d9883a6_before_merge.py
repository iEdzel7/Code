def ensure(module, base, state, names, autoremove):
    # Accumulate failures.  Package management modules install what they can
    # and fail with a message about what they can't.
    failures = []
    allow_erasing = False

    # Autoremove is called alone
    # Jump to remove path where base.autoremove() is run
    if not names and autoremove is not None:
        names = []
        state = 'absent'

    if names == ['*'] and state == 'latest':
        base.upgrade_all()
    else:
        pkg_specs, group_specs, filenames = _parse_spec_group_file(names)
        if group_specs:
            base.read_comps()

        pkg_specs = [p.strip() for p in pkg_specs]
        filenames = [f.strip() for f in filenames]
        groups = []
        environments = []
        for group_spec in (g.strip() for g in group_specs):
            group = base.comps.group_by_pattern(group_spec)
            if group:
                groups.append(group.id)
            else:
                environment = base.comps.environment_by_pattern(group_spec)
                if environment:
                    environments.append(environment.id)
                else:
                    module.fail_json(
                        msg="No group {0} available.".format(group_spec))

        if state in ['installed', 'present']:
            # Install files.
            _install_remote_rpms(base, filenames)

            # Install groups.
            for group in groups:
                try:
                    base.group_install(group, dnf.const.GROUP_PACKAGE_TYPES)
                except dnf.exceptions.Error as e:
                    # In dnf 2.0 if all the mandatory packages in a group do
                    # not install, an error is raised.  We want to capture
                    # this but still install as much as possible.
                    failures.append((group, to_native(e)))

            for environment in environments:
                try:
                    base.environment_install(environment, dnf.const.GROUP_PACKAGE_TYPES)
                except dnf.exceptions.Error as e:
                    failures.append((environment, to_native(e)))

            # Install packages.
            for pkg_spec in pkg_specs:
                _mark_package_install(module, base, pkg_spec)

        elif state == 'latest':
            # "latest" is same as "installed" for filenames.
            _install_remote_rpms(base, filenames)

            for group in groups:
                try:
                    try:
                        base.group_upgrade(group)
                    except dnf.exceptions.CompsError:
                        # If not already installed, try to install.
                        base.group_install(group, dnf.const.GROUP_PACKAGE_TYPES)
                except dnf.exceptions.Error as e:
                    failures.append((group, to_native(e)))

            for environment in environments:
                try:
                    try:
                        base.environment_upgrade(environment)
                    except dnf.exceptions.CompsError:
                        # If not already installed, try to install.
                        base.environment_install(environment, dnf.const.GROUP_PACKAGE_TYPES)
                except dnf.exceptions.Error as e:
                    failures.append((environment, to_native(e)))

            for pkg_spec in pkg_specs:
                # best effort causes to install the latest package
                # even if not previously installed
                base.conf.best = True
                base.install(pkg_spec)

        else:
            # state == absent
            if autoremove is not None:
                base.conf.clean_requirements_on_remove = autoremove

            if filenames:
                module.fail_json(
                    msg="Cannot remove paths -- please specify package name.")

            for group in groups:
                try:
                    base.group_remove(group)
                except dnf.exceptions.CompsError:
                    # Group is already uninstalled.
                    pass

            for environment in environments:
                try:
                    base.environment_remove(environment)
                except dnf.exceptions.CompsError:
                    # Environment is already uninstalled.
                    pass

            installed = base.sack.query().installed()
            for pkg_spec in pkg_specs:
                if installed.filter(name=pkg_spec):
                    base.remove(pkg_spec)

            # Like the dnf CLI we want to allow recursive removal of dependent
            # packages
            allow_erasing = True

            if autoremove:
                base.autoremove()

    if not base.resolve(allow_erasing=allow_erasing):
        if failures:
            module.fail_json(msg='Failed to install some of the '
                                 'specified packages',
                             failures=failures)
        module.exit_json(msg="Nothing to do")
    else:
        if module.check_mode:
            if failures:
                module.fail_json(msg='Failed to install some of the '
                                     'specified packages',
                                 failures=failures)
            module.exit_json(changed=True)

        base.download_packages(base.transaction.install_set)
        base.do_transaction()
        response = {'changed': True, 'results': []}
        for package in base.transaction.install_set:
            response['results'].append("Installed: {0}".format(package))
        for package in base.transaction.remove_set:
            response['results'].append("Removed: {0}".format(package))

        if failures:
            module.fail_json(msg='Failed to install some of the '
                                 'specified packages',
                             failures=failures)
        module.exit_json(**response)