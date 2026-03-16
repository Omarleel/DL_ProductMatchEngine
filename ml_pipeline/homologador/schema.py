class FeatureSchema2:
    TEXT_BASES = ("text", "base_text")
    CATEGORICAL_BASES = ("unit", "type")
    NUMERIC_BASES = ("cost", "peso", "factor", "content", "total")

    @staticmethod
    def fact(base: str) -> str:
        return f"fact_{base}"

    @staticmethod
    def master(base: str) -> str:
        return f"master_{base}"

    @classmethod
    def all_pair_columns(cls) -> tuple[str, ...]:
        cols = []
        for base in cls.TEXT_BASES + cls.CATEGORICAL_BASES + cls.NUMERIC_BASES:
            cols.extend([cls.fact(base), cls.master(base)])
        return tuple(cols)