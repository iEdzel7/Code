def _safe_ParseResult(parts, encoding='utf8', path_encoding='utf8'):
    return (
        to_native_str(parts.scheme),
        to_native_str(parts.netloc.encode('idna')),

        # default encoding for path component SHOULD be UTF-8
        quote(to_bytes(parts.path, path_encoding), _safe_chars),
        quote(to_bytes(parts.params, path_encoding), _safe_chars),

        # encoding of query and fragment follows page encoding
        # or form-charset (if known and passed)
        quote(to_bytes(parts.query, encoding), _safe_chars),
        quote(to_bytes(parts.fragment, encoding), _safe_chars)
    )