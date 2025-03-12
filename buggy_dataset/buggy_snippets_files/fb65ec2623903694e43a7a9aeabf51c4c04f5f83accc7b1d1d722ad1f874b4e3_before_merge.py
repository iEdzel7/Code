    def _edit_file_helper(self, filename, secret,
                          existing_data=None, force_save=False, vault_id=None):

        # Create a tempfile
        root, ext = os.path.splitext(os.path.realpath(filename))
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)

        try:
            if existing_data:
                self.write_data(existing_data, tmp_path, shred=False)

            # drop the user into an editor on the tmp file
            subprocess.call(self._editor_shell_command(tmp_path))
        except:
            # whatever happens, destroy the decrypted file
            self._shred_file(tmp_path)
            raise

        b_tmpdata = self.read_data(tmp_path)

        # Do nothing if the content has not changed
        if existing_data == b_tmpdata and not force_save:
            self._shred_file(tmp_path)
            return

        # encrypt new data and write out to tmp
        # An existing vaultfile will always be UTF-8,
        # so decode to unicode here
        b_ciphertext = self.vault.encrypt(b_tmpdata, secret, vault_id=vault_id)
        self.write_data(b_ciphertext, tmp_path)

        # shuffle tmp file into place
        self.shuffle_files(tmp_path, filename)
        display.vvvvv('Saved edited file "%s" encrypted using %s and  vault id "%s"' % (filename, secret, vault_id))