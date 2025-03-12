def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sort Python import definitions alphabetically "
        "within logical sections. Run with no arguments to run "
        "interactively. Run with `-` as the first argument to read from "
        "stdin. Otherwise provide a list of files to sort."
    )
    inline_args_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--src",
        "--src-path",
        dest="src_paths",
        action="append",
        help="Add an explicitly defined source path "
        "(modules within src paths have their imports automatically catorgorized as first_party).",
    )
    parser.add_argument(
        "-a",
        "--add-import",
        dest="add_imports",
        action="append",
        help="Adds the specified import line to all files, "
        "automatically determining correct placement.",
    )
    parser.add_argument(
        "--ac",
        "--atomic",
        dest="atomic",
        action="store_true",
        help="Ensures the output doesn't save if the resulting file contains syntax errors.",
    )
    parser.add_argument(
        "--af",
        "--force-adds",
        dest="force_adds",
        action="store_true",
        help="Forces import adds even if the original file is empty.",
    )
    parser.add_argument(
        "-b",
        "--builtin",
        dest="known_standard_library",
        action="append",
        help="Force isort to recognize a module as part of the python standard library.",
    )
    parser.add_argument(
        "-c",
        "--check-only",
        "--check",
        action="store_true",
        dest="check",
        help="Checks the file for unsorted / unformatted imports and prints them to the "
        "command line without modifying the file.",
    )
    parser.add_argument(
        "--ca",
        "--combine-as",
        dest="combine_as_imports",
        action="store_true",
        help="Combines as imports on the same line.",
    )
    parser.add_argument(
        "--cs",
        "--combine-star",
        dest="combine_star",
        action="store_true",
        help="Ensures that if a star import is present, "
        "nothing else is imported from that namespace.",
    )
    parser.add_argument(
        "-d",
        "--stdout",
        help="Force resulting output to stdout, instead of in-place.",
        dest="write_to_stdout",
        action="store_true",
    )
    parser.add_argument(
        "--df",
        "--diff",
        dest="show_diff",
        action="store_true",
        help="Prints a diff of all the changes isort would make to a file, instead of "
        "changing it in place",
    )
    parser.add_argument(
        "--ds",
        "--no-sections",
        help="Put all imports into the same section bucket",
        dest="no_sections",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--balanced",
        dest="balanced_wrapping",
        action="store_true",
        help="Balances wrapping to produce the most consistent line length possible",
    )
    parser.add_argument(
        "-f",
        "--future",
        dest="known_future_library",
        action="append",
        help="Force isort to recognize a module as part of the future compatibility libraries.",
    )
    parser.add_argument(
        "--fas",
        "--force-alphabetical-sort",
        action="store_true",
        dest="force_alphabetical_sort",
        help="Force all imports to be sorted as a single section",
    )
    parser.add_argument(
        "--fass",
        "--force-alphabetical-sort-within-sections",
        action="store_true",
        dest="force_alphabetical_sort",
        help="Force all imports to be sorted alphabetically within a section",
    )
    parser.add_argument(
        "--ff",
        "--from-first",
        dest="from_first",
        help="Switches the typical ordering preference, "
        "showing from imports first then straight ones.",
    )
    parser.add_argument(
        "--fgw",
        "--force-grid-wrap",
        nargs="?",
        const=2,
        type=int,
        dest="force_grid_wrap",
        help="Force number of from imports (defaults to 2) to be grid wrapped regardless of line "
        "length",
    )
    parser.add_argument(
        "--fss",
        "--force-sort-within-sections",
        action="store_true",
        dest="force_sort_within_sections",
        help="Force imports to be sorted by module, independent of import_type",
    )
    parser.add_argument(
        "-i",
        "--indent",
        help='String to place for indents defaults to "    " (4 spaces).',
        dest="indent",
        type=str,
    )
    parser.add_argument(
        "-j", "--jobs", help="Number of files to process in parallel.", dest="jobs", type=int
    )
    parser.add_argument(
        "-k",
        "--keep-direct-and-as",
        dest="keep_direct_and_as_imports",
        action="store_true",
        help="Turns off default behavior that removes direct imports when as imports exist.",
    )
    parser.add_argument("--lai", "--lines-after-imports", dest="lines_after_imports", type=int)
    parser.add_argument("--lbt", "--lines-between-types", dest="lines_between_types", type=int)
    parser.add_argument(
        "--le",
        "--line-ending",
        dest="line_ending",
        help="Forces line endings to the specified value. "
        "If not set, values will be guessed per-file.",
    )
    parser.add_argument(
        "--ls",
        "--length-sort",
        help="Sort imports by their string length.",
        dest="length_sort",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--multi-line",
        dest="multi_line_output",
        choices=list(WrapModes.__members__.keys())
        + [str(mode.value) for mode in WrapModes.__members__.values()],
        type=str,
        help="Multi line output (0-grid, 1-vertical, 2-hanging, 3-vert-hanging, 4-vert-grid, "
        "5-vert-grid-grouped, 6-vert-grid-grouped-no-comma).",
    )
    parser.add_argument(
        "-n",
        "--ensure-newline-before-comments",
        dest="ensure_newline_before_comments",
        action="store_true",
        help="Inserts a blank line before a comment following an import.",
    )
    inline_args_group.add_argument(
        "--nis",
        "--no-inline-sort",
        dest="no_inline_sort",
        action="store_true",
        help="Leaves `from` imports with multiple imports 'as-is' "
        "(e.g. `from foo import a, c ,b`).",
    )
    parser.add_argument(
        "--nlb",
        "--no-lines-before",
        help="Sections which should not be split with previous by empty lines",
        dest="no_lines_before",
        action="append",
    )
    parser.add_argument(
        "-o",
        "--thirdparty",
        dest="known_third_party",
        action="append",
        help="Force isort to recognize a module as being part of a third party library.",
    )
    parser.add_argument(
        "--ot",
        "--order-by-type",
        dest="order_by_type",
        action="store_true",
        help="Order imports by type in addition to alphabetically",
    )
    parser.add_argument(
        "--dt",
        "--dont-order-by-type",
        dest="dont_order_by_type",
        action="store_true",
        help="Don't order imports by type in addition to alphabetically",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="known_first_party",
        action="append",
        help="Force isort to recognize a module as being part of the current python project.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        dest="quiet",
        help="Shows extra quiet output, only errors are outputted.",
    )
    parser.add_argument(
        "--rm",
        "--remove-import",
        dest="remove_imports",
        action="append",
        help="Removes the specified import from all files.",
    )
    parser.add_argument(
        "--rr",
        "--reverse-relative",
        dest="reverse_relative",
        action="store_true",
        help="Reverse order of relative imports.",
    )
    parser.add_argument(
        "-s",
        "--skip",
        help="Files that sort imports should skip over. If you want to skip multiple "
        "files you should specify twice: --skip file1 --skip file2.",
        dest="skip",
        action="append",
    )
    parser.add_argument(
        "--sd",
        "--section-default",
        dest="default_section",
        help="Sets the default section for imports (by default FIRSTPARTY) options: "
        + str(sections.DEFAULT),
    )
    parser.add_argument(
        "--sg",
        "--skip-glob",
        help="Files that sort imports should skip over.",
        dest="skip_glob",
        action="append",
    )
    inline_args_group.add_argument(
        "--sl",
        "--force-single-line-imports",
        dest="force_single_line",
        action="store_true",
        help="Forces all from imports to appear on their own line",
    )
    parser.add_argument(
        "--nsl",
        "--single-line-exclusions",
        help="One or more modules to exclude from the single line rule.",
        dest="single_line_exclusions",
        action="append",
    )
    parser.add_argument(
        "--sp",
        "--settings-path",
        "--settings-file",
        "--settings",
        dest="settings_path",
        help="Explicitly set the settings path or file instead of auto determining "
        "based on file location.",
    )
    parser.add_argument(
        "-t",
        "--top",
        help="Force specific imports to the top of their appropriate section.",
        dest="force_to_top",
        action="append",
    )
    parser.add_argument(
        "--tc",
        "--trailing-comma",
        dest="include_trailing_comma",
        action="store_true",
        help="Includes a trailing comma on multi line imports that include parentheses.",
    )
    parser.add_argument(
        "--up",
        "--use-parentheses",
        dest="use_parentheses",
        action="store_true",
        help="Use parenthesis for line continuation on length limit instead of slashes.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        dest="show_version",
        help="Displays the currently installed version of isort.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Shows verbose output, such as when files are skipped or when a check is successful.",
    )
    parser.add_argument(
        "--virtual-env",
        dest="virtual_env",
        help="Virtual environment to use for determining whether a package is third-party",
    )
    parser.add_argument(
        "--conda-env",
        dest="conda_env",
        help="Conda environment to use for determining whether a package is third-party",
    )
    parser.add_argument(
        "--vn",
        "--version-number",
        action="version",
        version=__version__,
        help="Returns just the current version number without the logo",
    )
    parser.add_argument(
        "-l",
        "-w",
        "--line-length",
        "--line-width",
        help="The max length of an import line (used for wrapping long imports).",
        dest="line_length",
        type=int,
    )
    parser.add_argument(
        "--wl",
        "--wrap-length",
        dest="wrap_length",
        type=int,
        help="Specifies how long lines that are wrapped should be, if not set line_length is used."
        "\nNOTE: wrap_length must be LOWER than or equal to line_length.",
    )
    parser.add_argument(
        "--ws",
        "--ignore-whitespace",
        action="store_true",
        dest="ignore_whitespace",
        help="Tells isort to ignore whitespace differences when --check-only is being used.",
    )
    parser.add_argument(
        "--case-sensitive",
        dest="case_sensitive",
        action="store_true",
        help="Tells isort to include casing when sorting module names",
    )
    parser.add_argument(
        "--filter-files",
        dest="filter_files",
        action="store_true",
        help="Tells isort to filter files even when they are explicitly passed in as "
        "part of the command",
    )
    parser.add_argument(
        "files", nargs="*", help="One or more Python source files that need their imports sorted."
    )
    parser.add_argument(
        "--py",
        "--python-version",
        action="store",
        dest="py_version",
        choices=tuple(VALID_PY_TARGETS) + ("auto",),
        help="Tells isort to set the known standard library based on the the specified Python "
        "version. Default is to assume any Python 3 version could be the target, and use a union "
        "off all stdlib modules across versions. If auto is specified, the version of the "
        "interpreter used to run isort "
        f"(currently: {sys.version_info.major}{sys.version_info.minor}) will be used.",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        choices=list(profiles.keys()),
        type=str,
        help="Base profile type to use for configuration.",
    )
    parser.add_argument(
        "--interactive",
        dest="ask_to_apply",
        action="store_true",
        help="Tells isort to apply changes interactively.",
    )
    parser.add_argument(
        "--old-finders",
        "--magic",
        dest="old_finders",
        action="store_true",
        help="Use the old deprecated finder logic that relies on environment introspection magic.",
    )
    parser.add_argument(
        "--show-config",
        dest="show_config",
        action="store_true",
        help="See isort's determined config, as well as sources of config options.",
    )
    return parser