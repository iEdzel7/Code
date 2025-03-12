    def url_for(self, view_name, **kwargs):
        """
        Same as :func:`sanic.Sanic.url_for`, but automatically determine
        `scheme` and `netloc` base on the request. Since this method is aiming
        to generate correct schema & netloc, `_external` is implied.

        :param kwargs: takes same parameters as in :func:`sanic.Sanic.url_for`
        :return: an absolute url to the given view
        :rtype: str
        """
        # Full URL SERVER_NAME can only be handled in app.url_for
        if "//" in self.app.config.SERVER_NAME:
            return self.app.url_for(view_name, _external=True, **kwargs)

        scheme = self.scheme
        host = self.server_name
        port = self.server_port

        if (scheme.lower() in ("http", "ws") and port == 80) or (
            scheme.lower() in ("https", "wss") and port == 443
        ):
            netloc = host
        else:
            netloc = "{}:{}".format(host, port)

        return self.app.url_for(
            view_name, _external=True, _scheme=scheme, _server=netloc, **kwargs
        )