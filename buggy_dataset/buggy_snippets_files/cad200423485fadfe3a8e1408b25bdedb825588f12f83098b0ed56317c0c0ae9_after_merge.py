    def putmask(
        self, mask, new, inplace: bool = False, axis: int = 0, transpose: bool = False,
    ) -> List["Block"]:
        """
        See Block.putmask.__doc__
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        mask = _extract_bool_array(mask)

        new_values = self.values if inplace else self.values.copy()

        if isinstance(new, (np.ndarray, ExtensionArray)) and len(new) == len(mask):
            new = new[mask]

        mask = _safe_reshape(mask, new_values.shape)

        new_values[mask] = new
        return [self.make_block(values=new_values)]