    def run(self, tmp=None, task_vars=None):
        self._supports_check_mode = True
        self._supports_async = True

        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp  # tmp no longer has any effect

        category_names = self._task.args.get('category_names', [
            'CriticalUpdates',
            'SecurityUpdates',
            'UpdateRollups',
        ])
        if isinstance(category_names, AnsibleUnicode):
            category_names = [cat.strip() for cat in category_names.split(",")]

        state = self._task.args.get('state', 'installed')
        reboot = self._task.args.get('reboot', False)
        reboot_timeout = self._task.args.get('reboot_timeout',
                                             self.DEFAULT_REBOOT_TIMEOUT)
        use_task = boolean(self._task.args.get('use_scheduled_task', False),
                           strict=False)

        if state not in ['installed', 'searched']:
            result['failed'] = True
            result['msg'] = "state must be either installed or searched"
            return result

        try:
            reboot = boolean(reboot)
        except TypeError as exc:
            result['failed'] = True
            result['msg'] = "cannot parse reboot as a boolean: %s" % to_text(exc)
            return result

        if not isinstance(reboot_timeout, int):
            result['failed'] = True
            result['msg'] = "reboot_timeout must be an integer"
            return result

        if reboot and self._task.async_val > 0:
            result['failed'] = True
            result['msg'] = "async is not supported for this task when " \
                            "reboot=yes"
            return result

        # Run the module
        new_module_args = self._task.args.copy()
        new_module_args.pop('reboot', None)
        new_module_args.pop('reboot_timeout', None)
        result = self._run_win_updates(new_module_args, task_vars, use_task)

        # if the module failed to run at all then changed won't be populated
        # so we just return the result as is
        # https://github.com/ansible/ansible/issues/38232
        failed = result.get('failed', False)
        if ("updates" not in result.keys() and self._task.async_val == 0) or failed:
            result['failed'] = True
            return result

        changed = result.get('changed', False)
        updates = result.get('updates', dict())
        filtered_updates = result.get('filtered_updates', dict())
        found_update_count = result.get('found_update_count', 0)
        installed_update_count = result.get('installed_update_count', 0)

        # Handle automatic reboots if the reboot flag is set
        if reboot and state == 'installed' and not \
                self._play_context.check_mode:
            previously_errored = False
            while result['installed_update_count'] > 0 or \
                    result['found_update_count'] > 0 or \
                    result['reboot_required'] is True:
                display.vvv("win_updates: check win_updates results for "
                            "automatic reboot: %s" % json.dumps(result))

                # check if the module failed, break from the loop if it
                # previously failed and return error to the user
                if result.get('failed', False):
                    if previously_errored:
                        break
                    previously_errored = True
                else:
                    previously_errored = False

                reboot_error = None
                # check if a reboot was required before installing the updates
                if result.get('msg', '') == "A reboot is required before " \
                                            "more updates can be installed":
                    reboot_error = "reboot was required before more updates " \
                                   "can be installed"

                if result.get('reboot_required', False):
                    if reboot_error is None:
                        reboot_error = "reboot was required to finalise " \
                                       "update install"
                    try:
                        changed = True
                        self._reboot_server(task_vars, reboot_timeout,
                                            use_task)
                    except AnsibleError as exc:
                        result['failed'] = True
                        result['msg'] = "Failed to reboot remote host when " \
                                        "%s: %s" \
                                        % (reboot_error, to_text(exc))
                        break

                result.pop('msg', None)
                # rerun the win_updates module after the reboot is complete
                result = self._run_win_updates(new_module_args, task_vars,
                                               use_task)
                if result.get('failed', False):
                    return result

                result_updates = result.get('updates', dict())
                result_filtered_updates = result.get('filtered_updates', dict())
                updates = self._merge_dict(updates, result_updates)
                filtered_updates = self._merge_dict(filtered_updates,
                                                    result_filtered_updates)
                found_update_count += result.get('found_update_count', 0)
                installed_update_count += result.get('installed_update_count', 0)
                if result['changed']:
                    changed = True

        # finally create the return dict based on the aggregated execution
        # values if we are not in async
        if self._task.async_val == 0:
            result['changed'] = changed
            result['updates'] = updates
            result['filtered_updates'] = filtered_updates
            result['found_update_count'] = found_update_count
            result['installed_update_count'] = installed_update_count

        return result