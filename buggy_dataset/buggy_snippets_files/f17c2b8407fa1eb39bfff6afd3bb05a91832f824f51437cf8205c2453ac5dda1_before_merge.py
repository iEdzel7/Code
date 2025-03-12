def main():
    module = AnsibleModule(
        argument_spec=dict(
            state=dict(default='present', choices=['present', 'absent'], type='str'),
            name=dict(required=True, aliases=["src", "source"], type='str'),
            login_user=dict(default='guest', type='str'),
            login_password=dict(default='guest', type='str', no_log=True),
            login_host=dict(default='localhost', type='str'),
            login_port=dict(default='15672', type='str'),
            vhost=dict(default='/', type='str'),
            destination=dict(required=True, aliases=["dst", "dest"], type='str'),
            destination_type=dict(required=True, aliases=["type", "dest_type"], choices=["queue", "exchange"],
                                  type='str'),
            routing_key=dict(default='#', type='str'),
            arguments=dict(default=dict(), type='dict')
        ),
        supports_check_mode=True
    )

    if not HAS_REQUESTS:
        module.fail_json(msg="requests library is required for this module. To install, use `pip install requests`")

    RabbitMqBinding(module).run()