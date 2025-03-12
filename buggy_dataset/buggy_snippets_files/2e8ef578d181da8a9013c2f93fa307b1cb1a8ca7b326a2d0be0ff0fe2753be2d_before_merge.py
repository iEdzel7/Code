    def __init__(
        self, filepath_or_buffer, index=None, encoding="ISO-8859-1", chunksize=None
    ):

        self._encoding = encoding
        self._lines_read = 0
        self._index = index
        self._chunksize = chunksize

        if isinstance(filepath_or_buffer, str):
            (
                filepath_or_buffer,
                encoding,
                compression,
                should_close,
            ) = get_filepath_or_buffer(filepath_or_buffer, encoding=encoding)

        if isinstance(filepath_or_buffer, (str, bytes)):
            self.filepath_or_buffer = open(filepath_or_buffer, "rb")
        else:
            # Copy to BytesIO, and ensure no encoding
            contents = filepath_or_buffer.read()
            try:
                contents = contents.encode(self._encoding)
            except UnicodeEncodeError:
                pass
            self.filepath_or_buffer = BytesIO(contents)

        self._read_header()