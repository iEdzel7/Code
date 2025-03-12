def create_cloned_field(field: ModelField) -> ModelField:
    original_type = field.type_
    if is_dataclass(original_type) and hasattr(original_type, "__pydantic_model__"):
        original_type = original_type.__pydantic_model__  # type: ignore
    use_type = original_type
    if lenient_issubclass(original_type, BaseModel):
        original_type = cast(Type[BaseModel], original_type)
        use_type = create_model(original_type.__name__, __base__=original_type)
        for f in original_type.__fields__.values():
            use_type.__fields__[f.name] = create_cloned_field(f)

    new_field = create_response_field(name=field.name, type_=use_type)
    new_field.has_alias = field.has_alias
    new_field.alias = field.alias
    new_field.class_validators = field.class_validators
    new_field.default = field.default
    new_field.required = field.required
    new_field.model_config = field.model_config
    if PYDANTIC_1:
        new_field.field_info = field.field_info
    else:  # pragma: nocover
        new_field.schema = field.schema  # type: ignore
    new_field.allow_none = field.allow_none
    new_field.validate_always = field.validate_always
    if field.sub_fields:
        new_field.sub_fields = [
            create_cloned_field(sub_field) for sub_field in field.sub_fields
        ]
    if field.key_field:
        new_field.key_field = create_cloned_field(field.key_field)
    new_field.validators = field.validators
    if PYDANTIC_1:
        new_field.pre_validators = field.pre_validators
        new_field.post_validators = field.post_validators
    else:  # pragma: nocover
        new_field.whole_pre_validators = field.whole_pre_validators  # type: ignore
        new_field.whole_post_validators = field.whole_post_validators  # type: ignore
    new_field.parse_json = field.parse_json
    new_field.shape = field.shape
    try:
        new_field.populate_validators()
    except AttributeError:  # pragma: nocover
        # TODO: remove when removing support for Pydantic < 1.0.0
        new_field._populate_validators()  # type: ignore
    return new_field