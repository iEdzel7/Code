def run_and_report(func: Any) -> Any:
    try:
        return func()
    except Exception as ex:
        if _is_env_set("HYDRA_FULL_ERROR") or is_under_debugger():
            raise ex
        else:
            if isinstance(ex, CompactHydraException):
                sys.stderr.write(str(ex) + os.linesep)
                if isinstance(ex.__cause__, OmegaConfBaseException):
                    sys.stderr.write(str(ex.__cause__) + os.linesep)
            else:
                # Custom printing that strips the Hydra related stack frames from the top
                # And any omegaconf frames from the bottom.
                # It is possible to add additional libraries to sanitize from the bottom later,
                # maybe even make it configurable.
                tb: Any = ex.__traceback__
                search_max = 10
                # strip Hydra frames from start of stack
                # will strip until it hits run_job()
                while search_max > 0:
                    if tb is None:
                        break
                    frame = tb.tb_frame
                    tb = tb.tb_next
                    search_max = search_max - 1
                    if inspect.getframeinfo(frame).function == "run_job":
                        break

                if search_max == 0 or tb is None:
                    # could not detect run_job, probably a runtime exception before we got there.
                    # do not sanitize the stack trace.
                    print_exc()
                    sys.exit(1)

                # strip OmegaConf frames from bottom of stack
                end = tb
                num_frames = 0
                while end is not None:
                    frame = end.tb_frame
                    mdl = inspect.getmodule(frame)
                    assert mdl is not None
                    name = mdl.__name__
                    if name.startswith("omegaconf."):
                        break
                    end = end.tb_next
                    num_frames = num_frames + 1

                @dataclass
                class FakeTracebackType:
                    tb_next: Any = None  # Optional[FakeTracebackType]
                    tb_frame: Optional[FrameType] = None
                    tb_lasti: Optional[int] = None
                    tb_lineno: Optional[int] = None

                iter_tb = tb
                final_tb = FakeTracebackType()
                cur = final_tb
                added = 0
                while True:
                    cur.tb_lasti = iter_tb.tb_lasti
                    cur.tb_lineno = iter_tb.tb_lineno
                    cur.tb_frame = iter_tb.tb_frame

                    if added == num_frames - 1:
                        break
                    added = added + 1
                    cur.tb_next = FakeTracebackType()
                    cur = cur.tb_next
                    iter_tb = iter_tb.tb_next

                print_exception(etype=None, value=ex, tb=final_tb)  # type: ignore
            sys.stderr.write(
                "\nSet the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.\n"
            )
        sys.exit(1)