def static_analyzer(request, api=False):
    """Do static analysis on an request and save to db."""
    try:
        if api:
            typ = request.POST['scan_type']
            checksum = request.POST['hash']
            filename = request.POST['file_name']
            rescan = str(request.POST.get('re_scan', 0))
        else:
            typ = request.GET['type']
            checksum = request.GET['checksum']
            filename = request.GET['name']
            rescan = str(request.GET.get('rescan', 0))
        # Input validation
        app_dic = {}
        match = re.match('^[0-9a-f]{32}$', checksum)
        if (
                (
                    match
                ) and (
                    filename.lower().endswith('.apk')
                    or filename.lower().endswith('.zip')
                ) and (
                    typ in ['zip', 'apk']
                )
        ):
            app_dic['dir'] = Path(settings.BASE_DIR)  # BASE DIR
            app_dic['app_name'] = filename  # APP ORGINAL NAME
            app_dic['md5'] = checksum  # MD5
            # APP DIRECTORY
            app_dic['app_dir'] = Path(settings.UPLD_DIR) / checksum
            app_dic['tools_dir'] = app_dic['dir'] / 'StaticAnalyzer' / 'tools'
            app_dic['tools_dir'] = app_dic['tools_dir'].as_posix()
            logger.info('Starting Analysis on : %s', app_dic['app_name'])

            if typ == 'apk':
                app_dic['app_file'] = app_dic['md5'] + '.apk'  # NEW FILENAME
                app_dic['app_path'] = (
                    app_dic['app_dir'] / app_dic['app_file']).as_posix()
                app_dic['app_dir'] = app_dic['app_dir'].as_posix() + '/'
                # Check if in DB
                # pylint: disable=E1101
                db_entry = StaticAnalyzerAndroid.objects.filter(
                    MD5=app_dic['md5'])
                if db_entry.exists() and rescan == '0':
                    context = get_context_from_db_entry(db_entry)
                else:
                    # ANALYSIS BEGINS
                    app_dic['size'] = str(
                        file_size(app_dic['app_path'])) + 'MB'  # FILE SIZE
                    app_dic['sha1'], app_dic[
                        'sha256'] = hash_gen(app_dic['app_path'])
                    app_dic['files'] = unzip(
                        app_dic['app_path'], app_dic['app_dir'])
                    if not app_dic['files']:
                        # Can't Analyze APK, bail out.
                        msg = 'APK file is invalid or corrupt'
                        if api:
                            return print_n_send_error_response(
                                request,
                                msg,
                                True)
                        else:
                            return print_n_send_error_response(
                                request,
                                msg,
                                False)
                    app_dic['certz'] = get_hardcoded_cert_keystore(app_dic[
                                                                   'files'])

                    logger.info('APK Extracted')

                    # Manifest XML
                    app_dic['parsed_xml'] = get_manifest(
                        app_dic['app_path'],
                        app_dic['app_dir'],
                        app_dic['tools_dir'],
                        '',
                        True,
                    )

                    # get app_name
                    app_dic['real_name'] = get_app_name(
                        app_dic['app_path'],
                        app_dic['app_dir'],
                        app_dic['tools_dir'],
                        True,
                    )

                    # Get icon
                    res_path = os.path.join(app_dic['app_dir'], 'res')
                    app_dic['icon_hidden'] = True
                    # Even if the icon is hidden, try to guess it by the
                    # default paths
                    app_dic['icon_found'] = False
                    app_dic['icon_path'] = ''
                    # TODO: Check for possible different names for resource
                    # folder?
                    if os.path.exists(res_path):
                        icon_dic = get_icon(
                            app_dic['app_path'], res_path)
                        if icon_dic:
                            app_dic['icon_hidden'] = icon_dic['hidden']
                            app_dic['icon_found'] = bool(icon_dic['path'])
                            app_dic['icon_path'] = icon_dic['path']

                    # Set Manifest link
                    app_dic['mani'] = ('../ManifestView/?md5='
                                       + app_dic['md5']
                                       + '&type=apk&bin=1')
                    man_data_dic = manifest_data(app_dic['parsed_xml'])
                    app_dic['playstore'] = get_app_details(
                        man_data_dic['packagename'])
                    man_an_dic = manifest_analysis(
                        app_dic['parsed_xml'],
                        man_data_dic,
                        '',
                        app_dic['app_dir'],
                    )
                    bin_an_buff = []
                    bin_an_buff += elf_analysis(app_dic['app_dir'])
                    bin_an_buff += res_analysis(app_dic['app_dir'])
                    cert_dic = cert_info(
                        app_dic['app_dir'],
                        app_dic['app_file'])
                    apkid_results = apkid_analysis(app_dic[
                        'app_dir'], app_dic['app_path'], app_dic['app_name'])
                    tracker = Trackers.Trackers(
                        app_dic['app_dir'], app_dic['tools_dir'])
                    tracker_res = tracker.get_trackers()

                    apk_2_java(app_dic['app_path'], app_dic['app_dir'],
                               app_dic['tools_dir'])

                    dex_2_smali(app_dic['app_dir'], app_dic['tools_dir'])

                    code_an_dic = code_analysis(
                        app_dic['app_dir'],
                        'apk')

                    # Get the strings
                    string_res = strings_jar(
                        app_dic['app_file'],
                        app_dic['app_dir'])
                    if string_res:
                        app_dic['strings'] = string_res['strings']
                        app_dic['secrets'] = string_res['secrets']
                        code_an_dic['urls_list'].extend(
                            string_res['urls_list'])
                        code_an_dic['urls'].extend(string_res['url_nf'])
                        code_an_dic['emails'].extend(string_res['emails_nf'])
                    else:
                        app_dic['strings'] = []
                        app_dic['secrets'] = []
                    # Firebase DB Check
                    code_an_dic['firebase'] = firebase_analysis(
                        list(set(code_an_dic['urls_list'])))
                    # Domain Extraction and Malware Check
                    logger.info(
                        'Performing Malware Check on extracted Domains')
                    code_an_dic['domains'] = malware_check(
                        list(set(code_an_dic['urls_list'])))
                    # Copy App icon
                    copy_icon(app_dic['md5'], app_dic['icon_path'])
                    app_dic['zipped'] = 'apk'

                    logger.info('Connecting to Database')
                    try:
                        # SAVE TO DB
                        if rescan == '1':
                            logger.info('Updating Database...')
                            save_or_update(
                                'update',
                                app_dic,
                                man_data_dic,
                                man_an_dic,
                                code_an_dic,
                                cert_dic,
                                bin_an_buff,
                                apkid_results,
                                tracker_res,
                            )
                            update_scan_timestamp(app_dic['md5'])
                        elif rescan == '0':
                            logger.info('Saving to Database')
                            save_or_update(
                                'save',
                                app_dic,
                                man_data_dic,
                                man_an_dic,
                                code_an_dic,
                                cert_dic,
                                bin_an_buff,
                                apkid_results,
                                tracker_res,
                            )
                    except Exception:
                        logger.exception('Saving to Database Failed')
                    context = get_context_from_analysis(
                        app_dic,
                        man_data_dic,
                        man_an_dic,
                        code_an_dic,
                        cert_dic,
                        bin_an_buff,
                        apkid_results,
                        tracker_res,
                    )
                context['average_cvss'], context[
                    'security_score'] = score(context['code_analysis'])
                context['dynamic_analysis_done'] = is_file_exists(
                    os.path.join(app_dic['app_dir'], 'logcat.txt'))

                context['virus_total'] = None
                if settings.VT_ENABLED:
                    vt = VirusTotal.VirusTotal()
                    context['virus_total'] = vt.get_result(
                        app_dic['app_path'],
                        app_dic['md5'])
                template = 'static_analysis/android_binary_analysis.html'
                if api:
                    return context
                else:
                    return render(request, template, context)
            elif typ == 'zip':
                ios_ret = HttpResponseRedirect(
                    '/StaticAnalyzer_iOS/?name='
                    + app_dic['app_name']
                    + '&type=ios&checksum='
                    + app_dic['md5'])
                # Check if in DB
                # pylint: disable=E1101
                cert_dic = {
                    'certificate_info': '',
                    'certificate_status': '',
                    'description': '',
                }
                bin_an_buff = []
                app_dic['strings'] = []
                app_dic['secrets'] = []
                app_dic['zipped'] = ''
                # Above fields are only available for APK and not ZIP
                app_dic['app_file'] = app_dic['md5'] + '.zip'  # NEW FILENAME
                app_dic['app_path'] = (
                    app_dic['app_dir'] / app_dic['app_file']).as_posix()
                app_dic['app_dir'] = app_dic['app_dir'].as_posix() + '/'
                db_entry = StaticAnalyzerAndroid.objects.filter(
                    MD5=app_dic['md5'])
                ios_db_entry = StaticAnalyzerIOS.objects.filter(
                    MD5=app_dic['md5'])
                if db_entry.exists() and rescan == '0':
                    context = get_context_from_db_entry(db_entry)
                elif ios_db_entry.exists() and rescan == '0':
                    if api:
                        return {'type': 'ios'}
                    else:
                        return ios_ret
                else:
                    logger.info('Extracting ZIP')
                    app_dic['files'] = unzip(
                        app_dic['app_path'], app_dic['app_dir'])
                    # Check if Valid Directory Structure and get ZIP Type
                    pro_type, valid = valid_android_zip(app_dic['app_dir'])
                    if valid and pro_type == 'ios':
                        logger.info('Redirecting to iOS Source Code Analyzer')
                        if api:
                            return {'type': 'ios'}
                        else:
                            return ios_ret
                    app_dic['certz'] = get_hardcoded_cert_keystore(
                        app_dic['files'])
                    app_dic['zipped'] = pro_type
                    logger.info('ZIP Type - %s', pro_type)
                    if valid and (pro_type in ['eclipse', 'studio']):
                        # ANALYSIS BEGINS
                        app_dic['size'] = str(
                            file_size(app_dic['app_path'])) + 'MB'  # FILE SIZE
                        app_dic['sha1'], app_dic[
                            'sha256'] = hash_gen(app_dic['app_path'])

                        # Manifest XML
                        app_dic['persed_xml'] = get_manifest(
                            '',
                            app_dic['app_dir'],
                            app_dic['tools_dir'],
                            pro_type,
                            False,
                        )

                        # get app_name
                        app_dic['real_name'] = get_app_name(
                            app_dic['app_path'],
                            app_dic['app_dir'],
                            app_dic['tools_dir'],
                            False,
                        )

                        # Set manifest view link
                        app_dic['mani'] = (
                            '../ManifestView/?md5='
                            + app_dic['md5'] + '&type='
                            + pro_type + '&bin=0'
                        )

                        man_data_dic = manifest_data(app_dic['persed_xml'])
                        app_dic['playstore'] = get_app_details(
                            man_data_dic['packagename'])
                        man_an_dic = manifest_analysis(
                            app_dic['persed_xml'],
                            man_data_dic,
                            pro_type,
                            app_dic['app_dir'],
                        )
                        # Get icon
                        eclipse_res_path = os.path.join(
                            app_dic['app_dir'], 'res')
                        studio_res_path = os.path.join(
                            app_dic['app_dir'], 'app', 'src', 'main', 'res')
                        if os.path.exists(eclipse_res_path):
                            res_path = eclipse_res_path
                        elif os.path.exists(studio_res_path):
                            res_path = studio_res_path
                        else:
                            res_path = ''

                        app_dic['icon_hidden'] = man_an_dic['icon_hidden']
                        app_dic['icon_found'] = False
                        app_dic['icon_path'] = ''
                        if res_path:
                            app_dic['icon_path'] = find_icon_path_zip(
                                res_path, man_data_dic['icons'])
                            if app_dic['icon_path']:
                                app_dic['icon_found'] = True

                        if app_dic['icon_path']:
                            if os.path.exists(app_dic['icon_path']):
                                shutil.copy2(
                                    app_dic['icon_path'],
                                    os.path.join(
                                        settings.DWD_DIR,
                                        app_dic['md5'] + '-icon.png'))

                        code_an_dic = code_analysis(
                            app_dic['app_dir'],
                            pro_type)
                        # Firebase DB Check
                        code_an_dic['firebase'] = firebase_analysis(
                            list(set(code_an_dic['urls_list'])))
                        # Domain Extraction and Malware Check
                        logger.info(
                            'Performing Malware Check on extracted Domains')
                        code_an_dic['domains'] = malware_check(
                            list(set(code_an_dic['urls_list'])))
                        logger.info('Connecting to Database')
                        try:
                            # SAVE TO DB
                            if rescan == '1':
                                logger.info('Updating Database...')
                                save_or_update(
                                    'update',
                                    app_dic,
                                    man_data_dic,
                                    man_an_dic,
                                    code_an_dic,
                                    cert_dic,
                                    bin_an_buff,
                                    {},
                                    {},
                                )
                                update_scan_timestamp(app_dic['md5'])
                            elif rescan == '0':
                                logger.info('Saving to Database')
                                save_or_update(
                                    'save',
                                    app_dic,
                                    man_data_dic,
                                    man_an_dic,
                                    code_an_dic,
                                    cert_dic,
                                    bin_an_buff,
                                    {},
                                    {},
                                )
                        except Exception:
                            logger.exception('Saving to Database Failed')
                        context = get_context_from_analysis(
                            app_dic,
                            man_data_dic,
                            man_an_dic,
                            code_an_dic,
                            cert_dic,
                            bin_an_buff,
                            {},
                            {},
                        )
                    else:
                        msg = 'This ZIP Format is not supported'
                        if api:
                            return print_n_send_error_response(
                                request,
                                msg,
                                True)
                        else:
                            print_n_send_error_response(request, msg, False)
                            ctx = {
                                'title': 'Invalid ZIP archive',
                                'version': settings.MOBSF_VER,
                            }
                            template = 'general/zip.html'
                            return render(request, template, ctx)
                context['average_cvss'], context[
                    'security_score'] = score(context['code_analysis'])
                template = 'static_analysis/android_source_analysis.html'
                if api:
                    return context
                else:
                    return render(request, template, context)
            else:
                err = ('Only APK,IPA and Zipped '
                       'Android/iOS Source code supported now!')
                logger.error(err)
        else:
            msg = 'Hash match failed or Invalid file extension or file type'
            if api:
                return print_n_send_error_response(request, msg, True)
            else:
                return print_n_send_error_response(request, msg, False)

    except Exception as excep:
        logger.exception('Error Performing Static Analysis')
        msg = str(excep)
        exp = excep.__doc__
        if api:
            return print_n_send_error_response(request, msg, True, exp)
        else:
            return print_n_send_error_response(request, msg, False, exp)