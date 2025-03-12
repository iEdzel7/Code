def view_stat_map(stat_map_img, threshold=None, bg_img=None, vmax=None):
    """
    Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image. Should be 3D or
        4D with exactly one time point (i.e. stat_map_img.shape[-1] = 1)

    threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only values of amplitude above the
        given percentile will be shown.

    bg_img : Niimg-like object, optional (default=None)
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the stat map will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.

    vmax : float, optional (default=None)
        Upper bound for plotting

    Returns
    -------
    StatMapView : plot of the stat map.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook.

    """
    stat_map_img = check_niimg_3d(stat_map_img, dtype='auto')
    if bg_img is None:
        bg_img = datasets.load_mni152_template()
        bg_mask = datasets.load_mni152_brain_mask()
    else:
        bg_img = image.load_img(bg_img)
        bg_mask = image.new_img_like(bg_img, bg_img.get_data() != 0)
    stat_map_img = image.resample_to_img(stat_map_img, bg_img)
    stat_map_img = image.new_img_like(
        stat_map_img, stat_map_img.get_data() * bg_mask.get_data())
    if threshold is None:
        abs_threshold = 'null'
    else:
        abs_threshold = check_threshold(
            threshold, stat_map_img.get_data(), fast_abs_percentile)
        abs_threshold = str(abs_threshold)
    if vmax is None:
        vmax = np.abs(stat_map_img.get_data()).max()
    html = get_html_template('stat_map_template.html')
    html = html.replace('INSERT_STAT_MAP_DATA_HERE', _encode_nii(stat_map_img))
    html = html.replace('INSERT_MNI_DATA_HERE', _encode_nii(bg_img))
    html = html.replace('INSERT_ABS_MIN_HERE', abs_threshold)
    html = html.replace('INSERT_ABS_MAX_HERE', str(vmax))
    return StatMapView(html)