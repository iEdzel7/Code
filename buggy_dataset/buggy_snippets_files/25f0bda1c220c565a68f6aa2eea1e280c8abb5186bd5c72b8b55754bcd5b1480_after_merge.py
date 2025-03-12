def soiling_kimber(rainfall, cleaning_threshold=6, soiling_loss_rate=0.0015,
                   grace_period=14, max_soiling=0.3, manual_wash_dates=None,
                   initial_soiling=0, rain_accum_period=24):
    """
    Calculates fraction of energy lost due to soiling given rainfall data and
    daily loss rate using the Kimber model.

    Kimber soiling model [1]_ assumes soiling builds up at a daily rate unless
    the daily rainfall is greater than a threshold. The model also assumes that
    if daily rainfall has exceeded the threshold within a grace period, then
    the ground is too damp to cause soiling build-up. The model also assumes
    there is a maximum soiling build-up. Scheduled manual washes and rain
    events are assumed to reset soiling to zero.

    Parameters
    ----------
    rainfall: pandas.Series
        Accumulated rainfall at the end of each time period. [mm]
    cleaning_threshold: float, default 6
        Amount of daily rainfall required to clean the panels. [mm]
    soiling_loss_rate: float, default 0.0015
        Fraction of energy lost due to one day of soiling. [unitless]
    grace_period : int, default 14
        Number of days after a rainfall event when it's assumed the ground is
        damp, and so it's assumed there is no soiling. [days]
    max_soiling : float, default 0.3
        Maximum fraction of energy lost due to soiling. Soiling will build up
        until this value. [unitless]
    manual_wash_dates : sequence or None, default None
        List or tuple of dates as Python ``datetime.date`` when the panels were
        washed manually. Note there is no grace period after a manual wash, so
        soiling begins to build up immediately.
    initial_soiling : float, default 0
        Initial fraction of energy lost due to soiling at time zero in the
        `rainfall` series input. [unitless]
    rain_accum_period : int, default 24
        Period for accumulating rainfall to check against `cleaning_threshold`.
        The Kimber model defines this period as one day. [hours]

    Returns
    -------
    pandas.Series
        fraction of energy lost due to soiling, has same intervals as input

    Notes
    -----
    The soiling loss rate depends on both the geographical region and the
    soiling environment type. Rates measured by Kimber [1]_ are summarized in
    the following table:

    ===================  =======  =========  ======================
    Region/Environment   Rural    Suburban   Urban/Highway/Airport
    ===================  =======  =========  ======================
    Central Valley       0.0011   0.0019     0.0020
    Northern CA          0.0011   0.0010     0.0016
    Southern CA          0        0.0016     0.0019
    Desert               0.0030   0.0030     0.0030
    ===================  =======  =========  ======================

    Rainfall thresholds and grace periods may also vary by region. Please
    consult [1]_ for more information.

    References
    ----------
    .. [1] "The Effect of Soiling on Large Grid-Connected Photovoltaic Systems
       in California and the Southwest Region of the United States," Adrianne
       Kimber, et al., IEEE 4th World Conference on Photovoltaic Energy
       Conference, 2006, :doi:`10.1109/WCPEC.2006.279690`
    """
    # convert rain_accum_period to timedelta
    rain_accum_period = datetime.timedelta(hours=rain_accum_period)

    # convert grace_period to timedelta
    grace_period = datetime.timedelta(days=grace_period)

    # get indices as numpy datetime64, calculate timestep as numpy timedelta64,
    # and convert timestep to fraction of days
    rain_index_vals = rainfall.index.values
    timestep_interval = (rain_index_vals[1] - rain_index_vals[0])
    day_fraction = timestep_interval / np.timedelta64(24, 'h')

    # accumulate rainfall
    accumulated_rainfall = rainfall.rolling(
        rain_accum_period, closed='right').sum()

    # soiling rate
    soiling = np.ones_like(rainfall.values) * soiling_loss_rate * day_fraction
    soiling[0] = initial_soiling
    soiling = np.cumsum(soiling)
    soiling = pd.Series(soiling, index=rainfall.index, name='soiling')

    # rainfall events that clean the panels
    rain_events = accumulated_rainfall > cleaning_threshold

    # grace periods windows during which ground is assumed damp, so no soiling
    grace_windows = rain_events.rolling(grace_period, closed='right').sum() > 0

    # clean panels by subtracting soiling for indices in grace period windows
    cleaning = pd.Series(float('NaN'), index=rainfall.index)
    cleaning.iloc[0] = 0.0
    cleaning[grace_windows] = soiling[grace_windows]

    # manual wash dates
    if manual_wash_dates is not None:
        rain_tz = rainfall.index.tz
        # convert manual wash dates to datetime index in the timezone of rain
        manual_wash_dates = pd.DatetimeIndex(manual_wash_dates, tz=rain_tz)
        cleaning[manual_wash_dates] = soiling[manual_wash_dates]

    # remove soiling by foward filling cleaning where NaN
    soiling -= cleaning.ffill()

    # check if soiling has reached the maximum
    return soiling.where(soiling < max_soiling, max_soiling)