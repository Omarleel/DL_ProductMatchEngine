class AttributeSchemaV2:
    TEXT_COLUMNS = ("text", "base_text")
    CATEGORICAL_COLUMNS = (
        "provider",
        "unit",
        "type",
        "brand_hint",
        "category_hint",
    )
    NUMERIC_COLUMNS = ("cost", "factor", "content", "total", "peso")
    AUX_COLUMNS = (
        "brand_hint_present",
        "category_hint_present",
        "n_tokens",
        "n_chars",
        "has_digits",
        "has_pack_hint",
        "has_measure_hint",
        "pack_count_hint",
        "measure_value_hint",
    )
