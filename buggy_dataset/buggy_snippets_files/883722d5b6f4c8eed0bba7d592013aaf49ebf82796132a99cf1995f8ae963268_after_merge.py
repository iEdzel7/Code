def gen_python_files_in_dir(
    path: Path,
    root: Path,
    include: Pattern[str],
    exclude: Pattern[str],
    report: "Report",
) -> Iterator[Path]:
    """Generate all files under `path` whose paths are not excluded by the
    `exclude` regex, but are included by the `include` regex.

    Symbolic links pointing outside of the root directory are ignored.

    `report` is where output about exclusions goes.
    """
    assert root.is_absolute(), f"INTERNAL ERROR: `root` must be absolute but is {root}"
    for child in path.iterdir():
        try:
            normalized_path = "/" + child.resolve().relative_to(root).as_posix()
        except ValueError:
            if child.is_symlink():
                report.path_ignored(
                    child,
                    "is a symbolic link that points outside of the root directory",
                )
                continue

            raise

        if child.is_dir():
            normalized_path += "/"
        exclude_match = exclude.search(normalized_path)
        if exclude_match and exclude_match.group(0):
            report.path_ignored(child, f"matches the --exclude regular expression")
            continue

        if child.is_dir():
            yield from gen_python_files_in_dir(child, root, include, exclude, report)

        elif child.is_file():
            include_match = include.search(normalized_path)
            if include_match:
                yield child