    def format_meter(n, total, elapsed, ncols=None, prefix='',
                     ascii=False, unit='it', unit_scale=False, rate=None,
                     bar_format=None, postfix=None, unit_divisor=1000):
        """
        Return a string-based progress bar given some parameters

        Parameters
        ----------
        n  : int
            Number of finished iterations.
        total  : int
            The expected total number of iterations. If meaningless (), only
            basic progress statistics are displayed (no ETA).
        elapsed  : float
            Number of seconds passed since start.
        ncols  : int, optional
            The width of the entire output message. If specified,
            dynamically resizes the progress meter to stay within this bound
            [default: None]. The fallback meter width is 10 for the progress
            bar + no limit for the iterations counter and statistics. If 0,
            will not print any meter (only stats).
        prefix  : str, optional
            Prefix message (included in total width) [default: ''].
        ascii  : bool, optional
            If not set, use unicode (smooth blocks) to fill the meter
            [default: False]. The fallback is to use ASCII characters
            (1-9 #).
        unit  : str, optional
            The iteration unit [default: 'it'].
        unit_scale  : bool, optional
            If set, the number of iterations will printed with an
            appropriate SI metric prefix (K = 10^3, M = 10^6, etc.)
            [default: False].
        rate  : float, optional
            Manual override for iteration rate.
            If [default: None], uses n/elapsed.
        bar_format  : str, optional
            Specify a custom bar string formatting. May impact performance.
            [default: '{l_bar}{bar}{r_bar}'], where l_bar is
            '{desc}{percentage:3.0f}%|' and r_bar is
            '| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            Possible vars: bar, n, n_fmt, total, total_fmt, percentage,
            rate, rate_fmt, elapsed, remaining, l_bar, r_bar, desc.
        postfix  : str, optional
            Same as prefix but will be placed at the end as additional stats.
        unit_divisor  : float, optional
            [default: 1000], ignored unless `unit_scale` is True.

        Returns
        -------
        out  : Formatted meter and stats, ready to display.
        """

        # sanity check: total
        if total and n > total:
            total = None

        format_interval = tqdm.format_interval
        elapsed_str = format_interval(elapsed)

        # if unspecified, attempt to use rate = average speed
        # (we allow manual override since predicting time is an arcane art)
        if rate is None and elapsed:
            rate = n / elapsed
        inv_rate = 1 / rate if (rate and (rate < 1)) else None
        format_sizeof = tqdm.format_sizeof
        rate_fmt = ((format_sizeof(inv_rate if inv_rate else rate)
                    if unit_scale else
                    '{0:5.2f}'.format(inv_rate if inv_rate else rate))
                    if rate else '?') \
            + ('s' if inv_rate else unit) + '/' + (unit if inv_rate else 's')

        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = format_sizeof(total, divisor=unit_divisor) \
                if total else None
        else:
            n_fmt = str(n)
            total_fmt = str(total)

        # total is known: we can predict some stats
        if total:
            # fractional and percentage progress
            frac = n / total
            percentage = frac * 100

            remaining_str = format_interval((total - n) / rate) \
                if rate else '?'

            # format the stats displayed to the left and right sides of the bar
            l_bar = (prefix if prefix else '') + \
                '{0:3.0f}%|'.format(percentage)
            r_bar = '| {0}/{1} [{2}<{3}, {4}{5}]'.format(
                    n_fmt, total_fmt, elapsed_str, remaining_str, rate_fmt,
                    ', ' + postfix if postfix else '')

            if ncols == 0:
                return l_bar[:-1] + r_bar[1:]

            if bar_format:
                # Custom bar formatting
                # Populate a dict with all available progress indicators
                bar_args = {'n': n,
                            'n_fmt': n_fmt,
                            'total': total,
                            'total_fmt': total_fmt,
                            'percentage': percentage,
                            'rate': rate if inv_rate is None else inv_rate,
                            'rate_noinv': rate,
                            'rate_noinv_fmt': ((format_sizeof(rate)
                                               if unit_scale else
                                               '{0:5.2f}'.format(rate))
                                               if rate else '?') + unit + '/s',
                            'rate_fmt': rate_fmt,
                            'elapsed': elapsed_str,
                            'remaining': remaining_str,
                            'l_bar': l_bar,
                            'r_bar': r_bar,
                            'desc': prefix if prefix else '',
                            'postfix': ', ' + postfix if postfix else '',
                            # 'bar': full_bar  # replaced by procedure below
                            }

                # Interpolate supplied bar format with the dict
                if '{bar}' in bar_format:
                    # Format left/right sides of the bar, and format the bar
                    # later in the remaining space (avoid breaking display)
                    l_bar_user, r_bar_user = bar_format.split('{bar}')
                    l_bar = l_bar_user.format(**bar_args)
                    r_bar = r_bar_user.format(**bar_args)
                else:
                    # Else no progress bar, we can just format and return
                    return bar_format.format(**bar_args)

            # Formatting progress bar
            # space available for bar's display
            N_BARS = max(1, ncols - len(l_bar) - len(r_bar)) if ncols \
                else 10

            # format bar depending on availability of unicode/ascii chars
            if ascii:
                bar_length, frac_bar_length = divmod(
                    int(frac * N_BARS * 10), 10)

                bar = '#' * bar_length
                frac_bar = chr(48 + frac_bar_length) if frac_bar_length \
                    else ' '

            else:
                bar_length, frac_bar_length = divmod(int(frac * N_BARS * 8), 8)

                bar = _unich(0x2588) * bar_length
                frac_bar = _unich(0x2590 - frac_bar_length) \
                    if frac_bar_length else ' '

            # whitespace padding
            if bar_length < N_BARS:
                full_bar = bar + frac_bar + \
                    ' ' * max(N_BARS - bar_length - 1, 0)
            else:
                full_bar = bar + \
                    ' ' * max(N_BARS - bar_length, 0)

            # Piece together the bar parts
            return l_bar + full_bar + r_bar

        # no total: no progressbar, ETA, just progress stats
        else:
            return (prefix if prefix else '') + '{0}{1} [{2}, {3}{4}]'.format(
                n_fmt, unit, elapsed_str, rate_fmt,
                ', ' + postfix if postfix else '')