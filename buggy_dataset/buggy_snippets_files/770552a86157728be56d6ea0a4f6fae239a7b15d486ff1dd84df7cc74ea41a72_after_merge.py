def execute_show_command(command, module, command_type='cli_show'):
    command = {
        'command': command,
        'output': 'text',
    }

    return run_commands(module, command)