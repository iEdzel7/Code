def add_special_arithmetic_methods(cls):
    """
    Adds the full suite of special arithmetic methods (``__add__``,
    ``__sub__``, etc.) to the class.

    Parameters
    ----------
    cls : class
        special methods will be defined and pinned to this class
    """
    _, _, arith_method, comp_method, bool_method = _get_method_wrappers(cls)
    new_methods = _create_methods(cls, arith_method, comp_method, bool_method,
                                  special=True)
    # inplace operators (I feel like these should get passed an `inplace=True`
    # or just be removed

    def _wrap_inplace_method(method):
        """
        return an inplace wrapper for this method
        """

        def f(self, other):
            result = method(self, other)

            # this makes sure that we are aligned like the input
            # we are updating inplace so we want to ignore is_copy
            self._update_inplace(result.reindex_like(self, copy=False)._data,
                                 verify_is_copy=False)

            return self

        f.__name__ = "__i{name}__".format(name=method.__name__.strip("__"))
        return f

    new_methods.update(
        dict(__iadd__=_wrap_inplace_method(new_methods["__add__"]),
             __isub__=_wrap_inplace_method(new_methods["__sub__"]),
             __imul__=_wrap_inplace_method(new_methods["__mul__"]),
             __itruediv__=_wrap_inplace_method(new_methods["__truediv__"]),
             __ifloordiv__=_wrap_inplace_method(new_methods["__floordiv__"]),
             __imod__=_wrap_inplace_method(new_methods["__mod__"]),
             __ipow__=_wrap_inplace_method(new_methods["__pow__"])))
    if not compat.PY3:
        new_methods["__idiv__"] = _wrap_inplace_method(new_methods["__div__"])

    new_methods.update(
        dict(__iand__=_wrap_inplace_method(new_methods["__and__"]),
             __ior__=_wrap_inplace_method(new_methods["__or__"]),
             __ixor__=_wrap_inplace_method(new_methods["__xor__"])))

    add_methods(cls, new_methods=new_methods)