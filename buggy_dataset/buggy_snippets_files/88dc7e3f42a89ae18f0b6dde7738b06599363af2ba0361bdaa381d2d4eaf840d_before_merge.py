    def run(self, module, post_check):
        ''' Execute the configured source code in a module and run any post
        checks.

        Args:
            module (Module) : a module to execute the configured code in.

            post_check(callable) : a function that can raise an exception
                if expected post-conditions are not met after code execution.

        '''
        try:
            # Simulate the sys.path behaviour decribed here:
            #
            # https://docs.python.org/2/library/sys.html#sys.path
            _cwd = os.getcwd()
            _sys_path = list(sys.path)
            _sys_argv = list(sys.argv)
            sys.path.insert(0, os.path.dirname(self._path))
            sys.argv = [os.path.basename(self._path)] + self._argv

            exec(self._code, module.__dict__)
            post_check()

        except Exception as e:
            self._failed = True
            self._error_detail = traceback.format_exc()

            exc_type, exc_value, exc_traceback = sys.exc_info()
            filename, line_number, func, txt = traceback.extract_tb(exc_traceback)[-1]

            self._error = "%s\nFile \"%s\", line %d, in %s:\n%s" % (str(e), os.path.basename(filename), line_number, func, txt)

        finally:
            # undo sys.path, CWD fixups
            os.chdir(_cwd)
            sys.path = _sys_path
            sys.argv = _sys_argv
            self.ran = True