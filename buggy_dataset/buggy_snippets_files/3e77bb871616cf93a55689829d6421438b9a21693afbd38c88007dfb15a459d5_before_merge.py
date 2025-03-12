    def connect_to_influxdb(self):
        return InfluxDBClient(
            host=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database_name,
            ssl=self.params['ssl'],
            verify_ssl=self.params['validate_certs'],
            timeout=self.params['timeout'],
            retries=self.params['retries'],
            use_udp=self.params['use_udp'],
            udp_port=self.params['udp_port'],
            proxies=self.params['proxies'],
        )