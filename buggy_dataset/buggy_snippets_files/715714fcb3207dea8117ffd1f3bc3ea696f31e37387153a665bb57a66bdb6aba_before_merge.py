def add_file(dataset, registry, json_file, output_path):
    data_type = None
    line_count = None
    converted_path = None
    stdout = None
    link_data_only = dataset.get('link_data_only', 'copy_files')
    in_place = dataset.get('in_place', True)
    purge_source = dataset.get('purge_source', True)
    # in_place is True if there is no external chmod in place,
    # however there are other instances where modifications should not occur in_place:
    # in-place unpacking or editing of line-ending when linking in data or when
    # importing data from the FTP folder while purge_source is set to false
    if not purge_source and dataset.get('type') == 'ftp_import':
        # If we do not purge the source we should not modify it in place.
        in_place = False
    if dataset.type in ('server_dir', 'path_paste'):
        in_place = False
    check_content = dataset.get('check_content' , True)
    auto_decompress = dataset.get('auto_decompress', True)
    try:
        ext = dataset.file_type
    except AttributeError:
        file_err('Unable to process uploaded file, missing file_type parameter.', dataset, json_file)
        return

    if dataset.type == 'url':
        try:
            page = urlopen(dataset.path)  # page will be .close()ed by sniff methods
            temp_name, dataset.is_multi_byte = sniff.stream_to_file(page, prefix='url_paste', source_encoding=util.get_charset_from_http_headers(page.headers))
        except Exception as e:
            file_err('Unable to fetch %s\n%s' % (dataset.path, str(e)), dataset, json_file)
            return
        dataset.path = temp_name
    # See if we have an empty file
    if not os.path.exists(dataset.path):
        file_err('Uploaded temporary file (%s) does not exist.' % dataset.path, dataset, json_file)
        return
    if not os.path.getsize(dataset.path) > 0:
        file_err('The uploaded file is empty', dataset, json_file)
        return
    if not dataset.type == 'url':
        # Already set is_multi_byte above if type == 'url'
        try:
            dataset.is_multi_byte = multi_byte.is_multi_byte(codecs.open(dataset.path, 'r', 'utf-8').read(100))
        except UnicodeDecodeError as e:
            dataset.is_multi_byte = False
    # Is dataset an image?
    i_ext = get_image_ext(dataset.path)
    if i_ext:
        ext = i_ext
        data_type = ext
    # Is dataset content multi-byte?
    elif dataset.is_multi_byte:
        data_type = 'multi-byte char'
        ext = sniff.guess_ext(dataset.path, registry.sniff_order, is_multi_byte=True)
    # Is dataset content supported sniffable binary?
    else:
        # FIXME: This ignores the declared sniff order in datatype_conf.xml
        # resulting in improper behavior
        type_info = Binary.is_sniffable_binary(dataset.path)
        if type_info:
            data_type = type_info[0]
            ext = type_info[1]
    if not data_type:
        root_datatype = registry.get_datatype_by_extension(dataset.file_type)
        if getattr(root_datatype, 'compressed', False):
            data_type = 'compressed archive'
            ext = dataset.file_type
        else:
            # See if we have a gzipped file, which, if it passes our restrictions, we'll uncompress
            is_gzipped, is_valid = check_gzip(dataset.path, check_content=check_content)
            if is_gzipped and not is_valid:
                file_err('The gzipped uploaded file contains inappropriate content', dataset, json_file)
                return
            elif is_gzipped and is_valid and auto_decompress:
                if link_data_only == 'copy_files':
                    # We need to uncompress the temp_name file, but BAM files must remain compressed in the BGZF format
                    CHUNK_SIZE = 2 ** 20  # 1Mb
                    fd, uncompressed = tempfile.mkstemp(prefix='data_id_%s_upload_gunzip_' % dataset.dataset_id, dir=os.path.dirname(output_path), text=False)
                    gzipped_file = gzip.GzipFile(dataset.path, 'rb')
                    while 1:
                        try:
                            chunk = gzipped_file.read(CHUNK_SIZE)
                        except IOError:
                            os.close(fd)
                            os.remove(uncompressed)
                            file_err('Problem decompressing gzipped data', dataset, json_file)
                            return
                        if not chunk:
                            break
                        os.write(fd, chunk)
                    os.close(fd)
                    gzipped_file.close()
                    # Replace the gzipped file with the decompressed file if it's safe to do so
                    if not in_place:
                        dataset.path = uncompressed
                    else:
                        shutil.move(uncompressed, dataset.path)
                    os.chmod(dataset.path, 0o644)
                dataset.name = dataset.name.rstrip('.gz')
                data_type = 'gzip'
            if not data_type and bz2 is not None:
                # See if we have a bz2 file, much like gzip
                is_bzipped, is_valid = check_bz2(dataset.path, check_content)
                if is_bzipped and not is_valid:
                    file_err('The gzipped uploaded file contains inappropriate content', dataset, json_file)
                    return
                elif is_bzipped and is_valid and auto_decompress:
                    if link_data_only == 'copy_files':
                        # We need to uncompress the temp_name file
                        CHUNK_SIZE = 2 ** 20  # 1Mb
                        fd, uncompressed = tempfile.mkstemp(prefix='data_id_%s_upload_bunzip2_' % dataset.dataset_id, dir=os.path.dirname(output_path), text=False)
                        bzipped_file = bz2.BZ2File(dataset.path, 'rb')
                        while 1:
                            try:
                                chunk = bzipped_file.read(CHUNK_SIZE)
                            except IOError:
                                os.close(fd)
                                os.remove(uncompressed)
                                file_err('Problem decompressing bz2 compressed data', dataset, json_file)
                                return
                            if not chunk:
                                break
                            os.write(fd, chunk)
                        os.close(fd)
                        bzipped_file.close()
                        # Replace the bzipped file with the decompressed file if it's safe to do so
                        if not in_place:
                            dataset.path = uncompressed
                        else:
                            shutil.move(uncompressed, dataset.path)
                        os.chmod(dataset.path, 0o644)
                    dataset.name = dataset.name.rstrip('.bz2')
                    data_type = 'bz2'
            if not data_type:
                # See if we have a zip archive
                is_zipped = check_zip(dataset.path)
                if is_zipped and auto_decompress:
                    if link_data_only == 'copy_files':
                        CHUNK_SIZE = 2 ** 20  # 1Mb
                        uncompressed = None
                        uncompressed_name = None
                        unzipped = False
                        z = zipfile.ZipFile(dataset.path)
                        for name in z.namelist():
                            if name.endswith('/'):
                                continue
                            if unzipped:
                                stdout = 'ZIP file contained more than one file, only the first file was added to Galaxy.'
                                break
                            fd, uncompressed = tempfile.mkstemp(prefix='data_id_%s_upload_zip_' % dataset.dataset_id, dir=os.path.dirname(output_path), text=False)
                            if sys.version_info[:2] >= (2, 6):
                                zipped_file = z.open(name)
                                while 1:
                                    try:
                                        chunk = zipped_file.read(CHUNK_SIZE)
                                    except IOError:
                                        os.close(fd)
                                        os.remove(uncompressed)
                                        file_err('Problem decompressing zipped data', dataset, json_file)
                                        return
                                    if not chunk:
                                        break
                                    os.write(fd, chunk)
                                os.close(fd)
                                zipped_file.close()
                                uncompressed_name = name
                                unzipped = True
                            else:
                                # python < 2.5 doesn't have a way to read members in chunks(!)
                                try:
                                    outfile = open(uncompressed, 'wb')
                                    outfile.write(z.read(name))
                                    outfile.close()
                                    uncompressed_name = name
                                    unzipped = True
                                except IOError:
                                    os.close(fd)
                                    os.remove(uncompressed)
                                    file_err('Problem decompressing zipped data', dataset, json_file)
                                    return
                        z.close()
                        # Replace the zipped file with the decompressed file if it's safe to do so
                        if uncompressed is not None:
                            if not in_place:
                                dataset.path = uncompressed
                            else:
                                shutil.move(uncompressed, dataset.path)
                            os.chmod(dataset.path, 0o644)
                            dataset.name = uncompressed_name
                    data_type = 'zip'
            if not data_type:
                # TODO refactor this logic.  check_binary isn't guaranteed to be
                # correct since it only looks at whether the first 100 chars are
                # printable or not.  If someone specifies a known unsniffable
                # binary datatype and check_binary fails, the file gets mangled.
                if check_binary(dataset.path) or Binary.is_ext_unsniffable(dataset.file_type):
                    # We have a binary dataset, but it is not Bam, Sff or Pdf
                    data_type = 'binary'
                    # binary_ok = False
                    parts = dataset.name.split(".")
                    if len(parts) > 1:
                        ext = parts[-1].strip().lower()
                        if check_content and not Binary.is_ext_unsniffable(ext):
                            file_err('The uploaded binary file contains inappropriate content', dataset, json_file)
                            return
                        elif Binary.is_ext_unsniffable(ext) and dataset.file_type != ext:
                            err_msg = "You must manually set the 'File Format' to '%s' when uploading %s files." % (ext.capitalize(), ext)
                            file_err(err_msg, dataset, json_file)
                            return
            if not data_type:
                # We must have a text file
                if check_content and check_html(dataset.path):
                    file_err('The uploaded file contains inappropriate HTML content', dataset, json_file)
                    return
            if data_type != 'binary':
                if link_data_only == 'copy_files' and data_type not in ('gzip', 'bz2', 'zip'):
                    # Convert universal line endings to Posix line endings if to_posix_lines is True
                    # and the data is not binary or gzip-, bz2- or zip-compressed.
                    if dataset.to_posix_lines:
                        tmpdir = output_adjacent_tmpdir(output_path)
                        tmp_prefix = 'data_id_%s_convert_' % dataset.dataset_id
                        if dataset.space_to_tab:
                            line_count, converted_path = sniff.convert_newlines_sep2tabs(dataset.path, in_place=in_place, tmp_dir=tmpdir, tmp_prefix=tmp_prefix)
                        else:
                            line_count, converted_path = sniff.convert_newlines(dataset.path, in_place=in_place, tmp_dir=tmpdir, tmp_prefix=tmp_prefix)
                if dataset.file_type == 'auto':
                    ext = sniff.guess_ext(dataset.path, registry.sniff_order)
                else:
                    ext = dataset.file_type
                data_type = ext
    # Save job info for the framework
    if ext == 'auto' and data_type == 'binary':
        ext = 'data'
    if ext == 'auto' and dataset.ext:
        ext = dataset.ext
    if ext == 'auto':
        ext = 'data'
    datatype = registry.get_datatype_by_extension(ext)
    if dataset.type in ('server_dir', 'path_paste') and link_data_only == 'link_to_files':
        # Never alter a file that will not be copied to Galaxy's local file store.
        if datatype.dataset_content_needs_grooming(dataset.path):
            err_msg = 'The uploaded files need grooming, so change your <b>Copy data into Galaxy?</b> selection to be ' + \
                '<b>Copy files into Galaxy</b> instead of <b>Link to files without copying into Galaxy</b> so grooming can be performed.'
            file_err(err_msg, dataset, json_file)
            return
    if link_data_only == 'copy_files' and converted_path:
        # Move the dataset to its "real" path
        shutil.copy(converted_path, output_path)
        try:
            os.remove(converted_path)
        except Exception:
            pass
    elif link_data_only == 'copy_files':
        if purge_source:
            shutil.move(dataset.path, output_path)
        else:
            shutil.copy(dataset.path, output_path)
    # Write the job info
    stdout = stdout or 'uploaded %s file' % data_type
    info = dict(type='dataset',
                dataset_id=dataset.dataset_id,
                ext=ext,
                stdout=stdout,
                name=dataset.name,
                line_count=line_count)
    if dataset.get('uuid', None) is not None:
        info['uuid'] = dataset.get('uuid')
    json_file.write(dumps(info) + "\n")
    if link_data_only == 'copy_files' and datatype and datatype.dataset_content_needs_grooming(output_path):
        # Groom the dataset content if necessary
        datatype.groom_dataset_content(output_path)