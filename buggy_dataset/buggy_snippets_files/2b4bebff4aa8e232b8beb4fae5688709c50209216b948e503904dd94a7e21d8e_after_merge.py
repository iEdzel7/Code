    def get_real_file(self, file_path):
        """
        If the file is vault encrypted return a path to a temporary decrypted file
        If the file is not encrypted then the path is returned
        Temporary files are cleanup in the destructor
        """

        if not file_path or not isinstance(file_path, string_types):
            raise AnsibleParserError("Invalid filename: '%s'" % to_native(file_path))

        b_file_path = to_bytes(file_path, errors='surrogate_or_strict')
        if not self.path_exists(b_file_path) or not self.is_file(b_file_path):
            raise AnsibleFileNotFound("the file_name '%s' does not exist, or is not readable" % to_native(file_path))

        if not self._vault:
            self._vault = VaultLib(password="")

        real_path = self.path_dwim(file_path)

        try:
            with open(to_bytes(real_path), 'rb') as f:
                # Limit how much of the file is read since we do not know
                # whether this is a vault file and therefore it could be very
                # large.
                if is_encrypted_file(f, count=len(b_HEADER)):
                    # if the file is encrypted and no password was specified,
                    # the decrypt call would throw an error, but we check first
                    # since the decrypt function doesn't know the file name
                    data = f.read()
                    if not self._vault_password:
                        raise AnsibleParserError("A vault password must be specified to decrypt %s" % file_path)

                    data = self._vault.decrypt(data, filename=real_path)
                    # Make a temp file
                    real_path = self._create_content_tempfile(data)
                    self._tempfiles.add(real_path)

            return real_path

        except (IOError, OSError) as e:
            raise AnsibleParserError("an error occurred while trying to read the file '%s': %s" % (to_native(real_path), to_native(e)))