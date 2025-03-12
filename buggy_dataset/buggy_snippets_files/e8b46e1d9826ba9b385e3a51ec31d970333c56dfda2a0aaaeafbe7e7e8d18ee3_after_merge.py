def main(_):
  """Run the main test harness."""
  config_lib.CONFIG.AddContext(
      "AdminUI Context",
      "Context applied when running the admin user interface GUI.")
  startup.Init()

  if (not os.path.exists(os.path.join(config_lib.CONFIG[
      "AdminUI.document_root"], "dist/grr-ui.bundle.js")) or
      not os.path.exists(os.path.join(config_lib.CONFIG[
          "AdminUI.document_root"], "dist/grr-ui.bundle.css"))):
    raise RuntimeError("Can't find compiled JS/CSS bundles. "
                       "Please reinstall the PIP package using "
                       "\"pip install -e .\" to rebuild the bundles.")

  # Start up a server in another thread
  bind_address = config_lib.CONFIG["AdminUI.bind"]
  ip = ipaddr.IPAddress(bind_address)
  if ip.version == 4:
    # Address looks like an IPv4 address.
    ThreadingDjango.address_family = socket.AF_INET

  max_port = config_lib.CONFIG.Get("AdminUI.port_max",
                                   config_lib.CONFIG["AdminUI.port"])

  for port in range(config_lib.CONFIG["AdminUI.port"], max_port + 1):
    # Make a simple reference implementation WSGI server
    try:
      server = simple_server.make_server(bind_address,
                                         port,
                                         django_lib.GetWSGIHandler(),
                                         server_class=ThreadingDjango)
      break
    except socket.error as e:
      if e.errno == socket.errno.EADDRINUSE and port < max_port:
        logging.info("Port %s in use, trying %s", port, port + 1)
      else:
        raise

  proto = "HTTP"

  if config_lib.CONFIG["AdminUI.enable_ssl"]:
    cert_file = config_lib.CONFIG["AdminUI.ssl_cert_file"]
    if not cert_file:
      raise ValueError("Need a valid cert file to enable SSL.")

    key_file = config_lib.CONFIG["AdminUI.ssl_key_file"]
    server.socket = ssl.wrap_socket(server.socket,
                                    certfile=cert_file,
                                    keyfile=key_file,
                                    server_side=True)
    proto = "HTTPS"

    # SSL errors are swallowed by the WSGIServer so if your configuration does
    # not work, uncomment the line below, point your browser at the gui and look
    # at the log file to see why SSL complains:
    # server.socket.accept()

  sa = server.socket.getsockname()
  logging.info("Serving %s on %s port %d ...", proto, sa[0], sa[1])
  startup.DropPrivileges()

  server.serve_forever()