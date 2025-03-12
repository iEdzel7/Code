def merge(ctx, files, driver):
    """Copy valid pixels from input files to an output file.

    All files must have the same shape, number of bands, and data type.

    Input files are merged in their listed order using a reverse
    painter's algorithm.
    """
    import numpy as np

    verbosity = (ctx.obj and ctx.obj.get('verbosity')) or 1
    logger = logging.getLogger('rio')
    try:
        with rasterio.drivers(CPL_DEBUG=verbosity>2):
            output = files[-1]
            files = files[:-1]

            with rasterio.open(files[0]) as first:
                kwargs = first.meta
                kwargs['transform'] = kwargs.pop('affine')
                dest = np.zeros((first.count,) + first.shape, 
                    dtype=first.dtypes[0])

            nodataval = 0.0
            if os.path.exists(output):
                dst = rasterio.open(output, 'r+')
                nodataval = dst.nodatavals[0] or nodataval
            else:
                kwargs['driver'] == driver
                dst = rasterio.open(output, 'w', **kwargs)
                nodataval = first.nodatavals[0] or nodataval

            if nodataval:
                dest.fill(nodataval)

            for fname in reversed(files):
                with rasterio.open(fname) as src:
                    data = src.read()
                    try:
                        where = np.logical_and(
                                dest==nodataval, data.mask==False)
                    except AttributeError:
                        where = dest==nodataval
                    np.copyto(dest, data, where=where)

            if dst.mode == 'r+':
                data = dst.read()
                try:
                    where = np.logical_and(
                            dest==nodataval, data.mask==False)
                except AttributeError:
                    where = dest==nodataval
                np.copyto(dest, data, where=where)

            dst.write(dest)
            dst.close()

        sys.exit(0)
    except Exception:
        logger.exception("Failed. Exception caught")
        sys.exit(1)