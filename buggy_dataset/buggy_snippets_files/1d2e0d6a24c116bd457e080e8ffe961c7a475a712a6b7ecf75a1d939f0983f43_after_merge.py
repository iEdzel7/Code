    def _exec_jsonrpc(self, name, *args, **kwargs):

        req = request_builder(name, *args, **kwargs)
        reqid = req['id']

        if not os.path.exists(self.socket_path):
            raise ConnectionError('socket_path does not exist or cannot be found.'
                                  '\nSee the socket_path issue catergory in Network Debug and Troubleshooting Guide')

        try:
            data = json.dumps(req, cls=AnsibleJSONEncoder)
        except TypeError as exc:
            raise ConnectionError(
                "Failed to encode some variables as JSON for communication with ansible-connection. "
                "The original exception was: %s" % to_text(exc)
            )

        try:
            out = self.send(data)
            response = json.loads(out)

        except socket.error as e:
            raise ConnectionError('unable to connect to socket. See the socket_path issue catergory in Network Debug and Troubleshooting Guide',
                                  err=to_text(e, errors='surrogate_then_replace'), exception=traceback.format_exc())

        if response['id'] != reqid:
            raise ConnectionError('invalid json-rpc id received')

        return response