def get_argparser():
    """Get the argparse parser."""
    parser = argparse.ArgumentParser(prog='qutebrowser',
                                     description=qutebrowser.__description__)
    parser.add_argument('-B', '--basedir', help="Base directory for all "
                        "storage.")
    parser.add_argument('-C', '--config-py', help="Path to config.py.",
                        metavar='CONFIG')
    parser.add_argument('-V', '--version', help="Show version and quit.",
                        action='store_true')
    parser.add_argument('-s', '--set', help="Set a temporary setting for "
                        "this session.", nargs=2, action='append',
                        dest='temp_settings', default=[],
                        metavar=('OPTION', 'VALUE'))
    parser.add_argument('-r', '--restore', help="Restore a named session.",
                        dest='session')
    parser.add_argument('-R', '--override-restore', help="Don't restore a "
                        "session even if one would be restored.",
                        action='store_true')
    parser.add_argument('--target', choices=['auto', 'tab', 'tab-bg',
                                             'tab-silent', 'tab-bg-silent',
                                             'window', 'private-window'],
                        help="How URLs should be opened if there is already a "
                             "qutebrowser instance running.")
    parser.add_argument('--backend', choices=['webkit', 'webengine'],
                        help="Which backend to use.")

    parser.add_argument('--json-args', help=argparse.SUPPRESS)
    parser.add_argument('--temp-basedir-restarted', help=argparse.SUPPRESS)
    parser.add_argument('--desktop-file-name',
                        default="org.qutebrowser.qutebrowser",
                        help="Set the base name of the desktop entry for this "
                        "application. Used to set the app_id under Wayland. See "
                        "https://doc.qt.io/qt-5/qguiapplication.html#desktopFileName-prop")

    debug = parser.add_argument_group('debug arguments')
    debug.add_argument('-l', '--loglevel', dest='loglevel',
                       help="Override the configured console loglevel",
                       choices=['critical', 'error', 'warning', 'info',
                                'debug', 'vdebug'])
    debug.add_argument('--logfilter', type=logfilter_error,
                       help="Comma-separated list of things to be logged "
                       "to the debug log on stdout.")
    debug.add_argument('--loglines',
                       help="How many lines of the debug log to keep in RAM "
                       "(-1: unlimited).",
                       default=2000, type=int)
    debug.add_argument('-d', '--debug', help="Turn on debugging options.",
                       action='store_true')
    debug.add_argument('--json-logging', action='store_true', help="Output log"
                       " lines in JSON format (one object per line).")
    debug.add_argument('--nocolor', help="Turn off colored logging.",
                       action='store_false', dest='color')
    debug.add_argument('--force-color', help="Force colored logging",
                       action='store_true')
    debug.add_argument('--nowindow', action='store_true', help="Don't show "
                       "the main window.")
    debug.add_argument('-T', '--temp-basedir', action='store_true', help="Use "
                       "a temporary basedir.")
    debug.add_argument('--no-err-windows', action='store_true', help="Don't "
                       "show any error windows (used for tests/smoke.py).")
    debug.add_argument('--qt-arg', help="Pass an argument with a value to Qt. "
                       "For example, you can do "
                       "`--qt-arg geometry 650x555+200+300` to set the window "
                       "geometry.", nargs=2, metavar=('NAME', 'VALUE'),
                       action='append')
    debug.add_argument('--qt-flag', help="Pass an argument to Qt as flag.",
                       nargs=1, action='append')
    debug.add_argument('-D', '--debug-flag', type=debug_flag_error,
                       default=[], help="Pass name of debugging feature to be"
                       " turned on.", action='append', dest='debug_flags')
    parser.add_argument('command', nargs='*', help="Commands to execute on "
                        "startup.", metavar=':command')
    # URLs will actually be in command
    parser.add_argument('url', nargs='*', help="URLs to open on startup "
                        "(empty as a window separator).")
    return parser