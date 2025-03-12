def fetch_rpm_from_url(spec, module=None):
    # download package so that we can query it
    package_name, _ = os.path.splitext(str(spec.rsplit('/', 1)[1]))
    package_file = tempfile.NamedTemporaryFile(prefix=package_name, suffix='.rpm', delete=False)
    module.add_cleanup_file(package_file.name)
    try:
        rsp, info = fetch_url(module, spec)
        if not rsp:
            module.fail_json(msg="Failure downloading %s, %s" % (spec, info['msg']))
        data = rsp.read(BUFSIZE)
        while data:
            package_file.write(data)
            data = rsp.read(BUFSIZE)
        package_file.close()
    except Exception as e:
        if module:
            module.fail_json(msg="Failure downloading %s, %s" % (spec, to_native(e)))
        else:
            raise e

    return package_file.name