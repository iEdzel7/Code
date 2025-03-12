def calculate_multipart_etag(source_path, chunk_size=DEFAULT_CHUNK_SIZE):

    """
    calculates a multipart upload etag for amazon s3

    Arguments:

    source_path -- The file to calculate the etag for
    chunk_size -- The chunk size to calculate for.
    """

    md5s = []

    with open(source_path, 'rb') as fp:
        while True:

            data = fp.read(chunk_size)

            if not data:
                break
            md5s.append(hashlib.md5(data))

    if len(md5s) == 1:
        new_etag = '"{}"'.format(md5s[0].hexdigest())
    else:  # > 1
        digests = b"".join(m.digest() for m in md5s)

        new_md5 = hashlib.md5(digests)
        new_etag = '"{}-{}"'.format(new_md5.hexdigest(), len(md5s))

    return new_etag