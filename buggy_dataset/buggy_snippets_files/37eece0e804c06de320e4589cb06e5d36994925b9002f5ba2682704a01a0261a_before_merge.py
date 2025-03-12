    def query(
            self,
            querystr: str,
            param_types: Optional[Dict[str, Any]],
            param_values: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Queries The Graph for a particular query

        May raise:
        - RemoteError: If there is a problem querying
        """
        prefix = 'query '
        if param_types is not None:
            prefix += json.dumps(param_types).replace('"', '').replace('{', '(').replace('}', ')')
        prefix += '{'
        log.debug(f'Querying The Graph for {querystr}')
        try:
            result = self.client.execute(gql(prefix + querystr), variable_values=param_values)
        except requests.exceptions.RequestException:
            raise RemoteError(f'Failed to query the graph for {querystr}')

        log.debug('Got result from The Graph query')
        return result