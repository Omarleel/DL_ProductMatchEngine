from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CATEGORY_STOPWORDS = {
    "DE", "DEL", "LA", "EL", "LOS", "LAS", "Y", "EN", "CON", "SIN", "PARA",
    "UND", "UNID", "UNIDAD", "UNIDADES", "PACK", "PCK", "PQT", "PAQ", "CAJA",
    "CJA", "CJ", "BOT", "BOL", "LT", "L", "ML", "CC", "KG", "GR", "G",
    "FC", "CONT", "TIPO", "SIN_CATEGORIA",
}


@dataclass
class CategoryLexicon:
    category_to_terms: dict[str, list[str]]

    @staticmethod
    def _strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFKD", str(text))
        return "".join(ch for ch in text if not unicodedata.combining(ch))

    @classmethod
    def normalize_text(cls, text: str) -> str:
        if text is None:
            return ""
        s = cls._strip_accents(str(text)).upper().strip()
        s = re.sub(r"[^A-Z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        toks = cls.normalize_text(text).split()
        return [t for t in toks if len(t) > 2 and not t.isdigit() and t not in CATEGORY_STOPWORDS]

    @classmethod
    def _ngrams(cls, text: str) -> list[str]:
        toks = cls._tokenize(text)
        grams = list(toks)
        grams.extend([f"{a} {b}" for a, b in zip(toks, toks[1:])])
        return grams

    @classmethod
    def build(
        cls,
        df: pd.DataFrame,
        text_col: str = "base_text",
        label_col: str = "target_categoria",
        min_support: int = 4,
        top_k_per_category: int = 10,
    ) -> "CategoryLexicon":
        work = df[[text_col, label_col]].copy()
        work[text_col] = work[text_col].fillna("").astype(str)
        work[label_col] = work[label_col].fillna("SIN_CATEGORIA").astype(str)
        work = work[work[label_col] != "SIN_CATEGORIA"].copy()

        per_cat: dict[str, Counter] = defaultdict(Counter)
        global_counter: Counter = Counter()

        for row in work.itertuples(index=False):
            grams = cls._ngrams(getattr(row, text_col))
            if not grams:
                continue
            label = getattr(row, label_col)
            per_cat[label].update(grams)
            global_counter.update(set(grams))

        lexicon: dict[str, list[str]] = {}
        for cat, counter in per_cat.items():
            ranked: list[tuple[str, float]] = []
            for term, cat_count in counter.items():
                if cat_count < min_support:
                    continue
                precision = cat_count / max(global_counter[term], 1)
                score = precision * math.log1p(cat_count)
                if precision < 0.55:
                    continue
                ranked.append((term, score))
            ranked.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
            lexicon[cat] = [term for term, _ in ranked[:top_k_per_category]]
        return cls(category_to_terms=lexicon)

    def hits(self, text: str) -> list[str]:
        normalized = self.normalize_text(text)
        if not normalized:
            return []
        scored: list[tuple[str, int, int]] = []
        for cat, terms in self.category_to_terms.items():
            matches = [t for t in terms if re.search(rf"(?<![A-Z0-9]){re.escape(t)}(?![A-Z0-9])", normalized)]
            if matches:
                scored.append((cat, len(matches), max(len(t.split()) for t in matches)))
        scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return [cat for cat, _, _ in scored]

    def primary(self, text: str) -> str:
        hits = self.hits(text)
        return hits[0] if hits else ""

    def save(self, path: str | Path) -> None:
        with open(Path(path), "w", encoding="utf-8") as f:
            json.dump(self.category_to_terms, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CategoryLexicon":
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(category_to_terms={str(k): list(v) for k, v in data.items()})
