  def GetEntries(
      self, parser_mediator, key=None, registry_type=None, codepage='cp1252',
      **kwargs):
    """Collect Values under USBStor and return an event object for each one.

    Args:
      parser_mediator: A parser mediator object (instance of ParserMediator).
      key: Optional Registry key (instance of winreg.WinRegKey).
           The default is None.
      registry_type: Optional Registry type string. The default is None.
    """
    for subkey in key.GetSubkeys():
      text_dict = {}
      text_dict['subkey_name'] = subkey.name

      # Time last USB device of this class was first inserted.
      event_object = windows_events.WindowsRegistryEvent(
          subkey.last_written_timestamp, key.path, text_dict,
          usage=eventdata.EventTimestamp.FIRST_CONNECTED, offset=key.offset,
          registry_type=registry_type,
          source_append=': USBStor Entries')
      parser_mediator.ProduceEvent(event_object)

      # TODO: Determine if these 4 fields always exist.
      try:
        device_type, vendor, product, revision = subkey.name.split('&')
      except ValueError as exception:
        logging.warning(
            u'Unable to split string: {0:s} with error: {1:s}'.format(
                subkey.name, exception))

      text_dict['device_type'] = device_type
      text_dict['vendor'] = vendor
      text_dict['product'] = product
      text_dict['revision'] = revision

      for devicekey in subkey.GetSubkeys():
        text_dict['serial'] = devicekey.name

        friendly_name_value = devicekey.GetValue('FriendlyName')
        if friendly_name_value:
          text_dict['friendly_name'] = friendly_name_value.data
        else:
          text_dict.pop('friendly_name', None)

        # ParentIdPrefix applies to Windows XP Only.
        parent_id_prefix_value = devicekey.GetValue('ParentIdPrefix')
        if parent_id_prefix_value:
          text_dict['parent_id_prefix'] = parent_id_prefix_value.data
        else:
          text_dict.pop('parent_id_prefix', None)

        # Win7 - Last Connection.
        # Vista/XP - Time of an insert.
        event_object = windows_events.WindowsRegistryEvent(
            devicekey.last_written_timestamp, key.path, text_dict,
            usage=eventdata.EventTimestamp.LAST_CONNECTED, offset=key.offset,
            registry_type=registry_type,
            source_append=': USBStor Entries')
        parser_mediator.ProduceEvent(event_object)

        # Build list of first Insertion times.
        first_insert = []
        device_parameter_key = devicekey.GetSubkey('Device Parameters')
        if device_parameter_key:
          first_insert.append(device_parameter_key.last_written_timestamp)

        log_configuration_key = devicekey.GetSubkey('LogConf')
        if (log_configuration_key and
            log_configuration_key.last_written_timestamp not in first_insert):
          first_insert.append(log_configuration_key.last_written_timestamp)

        properties_key = devicekey.GetSubkey('Properties')
        if (properties_key and
            properties_key.last_written_timestamp not in first_insert):
          first_insert.append(properties_key.last_written_timestamp)

        # Add first Insertion times.
        for timestamp in first_insert:
          event_object = windows_events.WindowsRegistryEvent(
              timestamp, key.path, text_dict,
              usage=eventdata.EventTimestamp.LAST_CONNECTED, offset=key.offset,
              registry_type=registry_type,
              source_append=': USBStor Entries')
          parser_mediator.ProduceEvent(event_object)