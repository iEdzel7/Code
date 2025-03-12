    def edit_file(self, filename):
        vault_id_used = None
        vault_secret_used = None
        # follow the symlink
        filename = self._real_path(filename)

        b_vaulttext = self.read_data(filename)

        # vault or yaml files are always utf8
        vaulttext = to_text(b_vaulttext)

        try:
            # vaulttext gets converted back to bytes, but alas
            # TODO: return the vault_id that worked?
            plaintext, vault_id_used, vault_secret_used = self.vault.decrypt_and_get_vault_id(vaulttext)
        except AnsibleError as e:
            raise AnsibleError("%s for %s" % (to_bytes(e), to_bytes(filename)))

        # Figure out the vault id from the file, to select the right secret to re-encrypt it
        # (duplicates parts of decrypt, but alas...)
        dummy, dummy, cipher_name, vault_id = parse_vaulttext_envelope(b_vaulttext,
                                                                       filename=filename)

        # vault id here may not be the vault id actually used for decrypting
        # as when the edited file has no vault-id but is decrypted by non-default id in secrets
        # (vault_id=default, while a different vault-id decrypted)

        # Keep the same vault-id (and version) as in the header
        if cipher_name not in CIPHER_WRITE_WHITELIST:
            # we want to get rid of files encrypted with the AES cipher
            self._edit_file_helper(filename, vault_secret_used, existing_data=plaintext,
                                   force_save=True, vault_id=vault_id)
        else:
            self._edit_file_helper(filename, vault_secret_used, existing_data=plaintext,
                                   force_save=False, vault_id=vault_id)