import html
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


PACK_ALIASES = {
    "UND": "UND",
    "UNDS": "UND",
    "UNID": "UND",
    "UNIDADES": "UND",
    "UN": "UND",
    "U": "UND",
    "IT": "UND",
    "UAD": "UND",

    "PQT": "PQT",
    "PAQ": "PQT",
    "PAQT": "PQT",
    "PACK": "PCK",
    "PCK": "PCK",
    "PK": "PCK",

    "BOL": "BOL",
    "BLS": "BOL",

    "BOT": "BOT",
    "BTL": "BOT",

    "CJ": "CJA",
    "CAJ": "CJA",
    "CJA": "CJA",
    "CAJA": "CJA",

    "BDJ": "BDJ",
    "BAN": "BAN",

    "DP": "DP",
    "DISPLAY": "DP",
    "DISPLEY": "DP",

    "SOB": "SOB",
    "SACHET": "SACHET",
    "SACH": "SACHET",
    "ROL": "ROL",
    "TIRA": "TIRA",
}

MEASURE_ALIASES = {
    "MG": "MG",
    "GR": "GR",
    "G": "G",
    "KG": "KG",
    "KGS": "KG",
    "ML": "ML",
    "CC": "CC",
    "LT": "LT",
    "LTS": "LT",
    "LTR": "LT",
    "L": "L",
    "LB": "LB",
    "LIBRAS": "LB",
    "LIBRA": "LB",
    "ONZA": "OZ",
    "ONZAS": "OZ",
}

DIM_ALIASES = {
    "CM": "CM",
    "MM": "MM",
    "MPLG": "MPLG",
    "PLG": "PLG",
    "PULG": "PLG",
    "INCH": "INCH",
    "MTS": "MTS",
    "MTR": "MTR",
    "METRO": "METRO",
    "HJ": "HJ",
}

PACK_TOKENS = set(PACK_ALIASES.values())
MEASURE_TOKENS = set(MEASURE_ALIASES.values())
DIMENSION_TOKENS = set(DIM_ALIASES.values())

NOISE_PHRASES = [
    "POR ANTICIPO DE MERCADERIA",
]

# Tokens que sí conviene quitar del base norm, no del limpio.
NOISE_TOKENS = {
    "RN", "NF", "PE",
}


def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def renombrar_columnas_equivalentes(df: pd.DataFrame, equivalencias: dict) -> pd.DataFrame:
    df = limpiar_nombres_columnas(df)

    columnas_lower = {c.lower(): c for c in df.columns}
    renombres = {}

    for canonica, aliases in equivalencias.items():
        encontrada = None

        for nombre in [canonica] + aliases:
            key = nombre.lower().strip().replace("\ufeff", "")
            if key in columnas_lower:
                encontrada = columnas_lower[key]
                break

        if encontrada is not None and encontrada != canonica:
            renombres[encontrada] = canonica

    if renombres:
        df = df.rename(columns=renombres)

    return df


def validar_columnas(df: pd.DataFrame, requeridas: list, nombre_df: str) -> None:
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"En {nombre_df} faltan columnas requeridas: {faltantes}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


def quitar_acentos(texto: str) -> str:
    if pd.isna(texto):
        return ""
    texto = str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _prelimpiar_texto(texto: str) -> str:
    if pd.isna(texto):
        return ""

    texto = html.unescape(str(texto))
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = quitar_acentos(texto).upper()

    for frase in NOISE_PHRASES:
        texto = texto.replace(frase, " ")

    # Basura típica incrustada
    texto = re.sub(r"@#@.*", " ", texto)
    texto = re.sub(r"\[[^\s]*", " ", texto)

    # Separadores obvios
    texto = texto.replace("+", " ")
    texto = texto.replace("*", " ")
    texto = texto.replace("|", " ")
    texto = texto.replace("_", " ")
    texto = texto.replace(",", " ")
    texto = texto.replace(";", " ")
    texto = texto.replace("'", " ")
    texto = texto.replace('"', " ")
    texto = texto.replace("(", " ")
    texto = texto.replace(")", " ")
    texto = texto.replace("{", " ")
    texto = texto.replace("}", " ")

    # Mantener slash y guion internos, pero limpiar el resto del ruido
    texto = re.sub(r"[^\w\s/\-\.]", " ", texto)

    # Puntos como separadores, preservando decimales
    texto = re.sub(r"(?<=[A-Z])\.(?=[A-Z])", " ", texto)
    texto = re.sub(r"(?<=[A-Z])\.(?=\d)", " ", texto)
    texto = re.sub(r"(?<=\d)\.(?=[A-Z])", " ", texto)
    texto = re.sub(r"(?<!\d)\.(?!\d)", " ", texto)

    # Normalizar espacios
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _canonicalizar_token(token: str) -> str:
    t = token.strip().upper()

    if t in PACK_ALIASES:
        return PACK_ALIASES[t]
    if t in MEASURE_ALIASES:
        return MEASURE_ALIASES[t]
    if t in DIM_ALIASES:
        return DIM_ALIASES[t]
    return t


def _split_alpha_segment(seg: str) -> List[str]:
    seg = seg.strip().strip(".")
    if not seg:
        return []

    canon = _canonicalizar_token(seg)
    if canon != seg or seg in PACK_TOKENS or seg in MEASURE_TOKENS or seg in DIMENSION_TOKENS:
        return [canon]

    # Casos tipo CJAX / DPX / UX / UNDX / GRX
    if seg.endswith("X") and len(seg) > 1:
        left = seg[:-1]
        left_canon = _canonicalizar_token(left)
        if left_canon in PACK_TOKENS or left_canon in MEASURE_TOKENS or left_canon in DIMENSION_TOKENS:
            return [left_canon, "X"]

    # Casos tipo XUND / XCJA
    if seg.startswith("X") and len(seg) > 1:
        right = seg[1:]
        right_canon = _canonicalizar_token(right)
        if right_canon in PACK_TOKENS or right_canon in MEASURE_TOKENS or right_canon in DIMENSION_TOKENS:
            return ["X", right_canon]

    return [seg]


def _segmentar_chunk(chunk: str) -> List[str]:
    chunk = chunk.strip().strip(".")
    if not chunk or chunk == "-":
        return []

    # Preservar tokens compuestos útiles
    if re.fullmatch(r"\d+-\d+", chunk):
        return [chunk]

    # Preservar prefijos y tokens compuestos tipo:
    # 029-MACA, PE-MUESTRA, C/IMPR
    if re.fullmatch(r"[A-Z0-9]+(?:[-/][A-Z0-9]+)+", chunk):
        return [chunk]

    out: List[str] = []
    i = 0
    n = len(chunk)

    while i < n:
        ch = chunk[i]

        if ch.isspace():
            i += 1
            continue

        if ch.isdigit():
            j = i + 1
            while j < n:
                if chunk[j].isdigit():
                    j += 1
                    continue
                if chunk[j] == "." and j + 1 < n and chunk[j + 1].isdigit():
                    j += 1
                    continue
                break
            out.append(chunk[i:j].strip("."))
            i = j
            continue

        if ch == "X":
            out.append("X")
            i += 1
            continue

        if ch.isalpha():
            j = i + 1
            while j < n and chunk[j].isalpha():
                j += 1

            seg = chunk[i:j]
            out.extend(_split_alpha_segment(seg))
            i = j
            continue

        if ch in {"/", "-"}:
            # si quedó suelto, lo tratamos como separador
            i += 1
            continue

        i += 1

    return [t for t in out if t]


def _tokenizar_extraccion(texto: str) -> List[str]:
    bruto = _prelimpiar_texto(texto)
    if not bruto:
        return []

    # Espacios útiles entre letras/números.
    # Esto sí ayuda para LUMCARB500ML, 6UND1KG, etc.
    bruto = re.sub(r"(?<=\d)(?=[A-Z])", " ", bruto)
    bruto = re.sub(r"(?<=[A-Z])(?=\d)", " ", bruto)

    # Separar X solo cuando está entre números
    bruto = re.sub(r"(?<=\d)[Xx](?=\d)", " X ", bruto)

    # Separar X cuando viene pegada a unidades o empaques compactos
    bruto = re.sub(r"\b(CJA|CJ|CAJ|CAJA|UND|UNID|UNIDADES|UN|U|IT|PQT|PAQ|PAQT|PACK|PCK|PK|BOL|BLS|BOT|BTL|BDJ|BAN|DP|DISPLAY|DISPLEY|SOB|SACHET|SACH|ROL|TIRA)X(?=\d)", r"\1 X ", bruto)
    bruto = re.sub(r"(?<=\d)X(?=(CJA|CJ|CAJ|CAJA|UND|UNID|UNIDADES|UN|U|IT|PQT|PAQ|PAQT|PACK|PCK|PK|BOL|BLS|BOT|BTL|BDJ|BAN|DP|DISPLAY|DISPLEY|SOB|SACHET|SACH|ROL|TIRA)\b)", " X ", bruto)

    bruto = re.sub(r"\s+", " ", bruto).strip()

    tokens: List[str] = []
    for chunk in bruto.split():
        tokens.extend(_segmentar_chunk(chunk))

    # Canonicalizar una última vez
    tokens = [_canonicalizar_token(t) for t in tokens if t]

    # Limpiar ruido residual
    tokens = [t for t in tokens if t not in {"", ".", "-"}]
    return tokens


def limpiar_descripcion_bruta(texto: str) -> str:
    return " ".join(_tokenizar_extraccion(texto)).strip()


def normalizar_texto(texto: str) -> str:
    return limpiar_descripcion_bruta(texto)


def normalizar_codigo(codigo: str) -> str:
    if pd.isna(codigo):
        return ""
    codigo = str(codigo).strip().upper()
    codigo = re.sub(r"\s+", "", codigo)
    return codigo


def normalizar_unidad(u: str) -> str:
    u = normalizar_texto(u)
    mapa = {
        "UNIDAD PIEZA": "UNIDAD",
        "UNIDAD PIEZAS": "UNIDAD",
        "UND": "UNIDAD",
        "PIEZA": "UNIDAD",
        "UND PIEZA": "UNIDAD",
        "CJ": "CAJA",
        "CAJ": "CAJA",
        "CJA": "CAJA",
    }
    return mapa.get(u, u)


def _token_numero(token: str) -> Optional[float]:
    token = token.strip().strip(".")
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    if re.fullmatch(r"\d+/\d+", token):
        num, den = token.split("/")
        return float(num) / float(den) if float(den) != 0 else None
    return None


def _token_word(token: str) -> str:
    return _canonicalizar_token(token.strip().strip(".,;:").upper())


def _convertir_unidad_contenido(valor: float, unidad: str) -> Tuple[float, str]:
    unidad = unidad.upper()
    if unidad == "MG":
        return valor / 1000.0, "MASS"
    if unidad in {"GR", "G"}:
        return valor, "MASS"
    if unidad == "KG":
        return valor * 1000.0, "MASS"
    if unidad == "LB":
        return valor * 453.592, "MASS" 
    if unidad == "OZ":
        return valor * 28.3495, "MASS"
    if unidad in {"ML", "CC"}:
        return valor, "VOLUME"
    if unidad in {"LT", "L"}:
        return valor * 1000.0, "VOLUME"
    return 0.0, "NONE"


def _extraer_medida(tokens: List[str]) -> Tuple[float, str, int]:
    for i in range(len(tokens) - 1):
        num = _token_numero(tokens[i])
        unit = _token_word(tokens[i + 1])

        if num is None or unit not in MEASURE_TOKENS:
            continue

        valor, tipo = _convertir_unidad_contenido(num, unit)
        if valor > 0:
            return valor, tipo, i

    return 0.0, "NONE", -1


def _extraer_factores(tokens: List[str], measure_idx: int) -> Tuple[List[int], Set[Any]]:
    counts: List[int] = []
    usados: Set[Any] = set()

    def add_count(n: int, idxs: List[Any]) -> None:
        if n <= 1:
            return
        if any(i in usados for i in idxs):
            return
        counts.append(int(n))
        usados.update(idxs)

    # 1) Patrones tipo 8 PQT / 12 BOT / 50 UND
    i = 0
    while i < len(tokens) - 1:
        num = _token_numero(tokens[i])
        tok = _token_word(tokens[i + 1])

        if num is not None and float(num).is_integer() and tok in PACK_TOKENS:
            add_count(int(num), [i])
            i += 2
            continue

        # 2) Patrones tipo BOL X 80 / CJA X 24
        tok0 = _token_word(tokens[i])
        tok1 = _token_word(tokens[i + 1])

        if tok0 in PACK_TOKENS and tok1 == "X" and i + 2 < len(tokens):
            num2 = _token_numero(tokens[i + 2])
            if num2 is not None and float(num2).is_integer():
                add_count(int(num2), [i + 2])
                i += 3
                continue

        # 3) Patrones tipo 16-6 PCK
        if re.fullmatch(r"\d+-\d+", _token_word(tokens[i])) and tok in PACK_TOKENS:
            a, b = _token_word(tokens[i]).split("-")
            add_count(int(a), [f"{i}:a"])
            add_count(int(b), [f"{i}:b"])
            i += 2
            continue

        i += 1

    # 4) Patrones tipo 12 X 270 GR
    for i in range(len(tokens) - 3):
        a = _token_numero(tokens[i])
        b = _token_numero(tokens[i + 2])

        if (
            a is not None
            and float(a).is_integer()
            and _token_word(tokens[i + 1]) == "X"
            and b is not None
            and _token_word(tokens[i + 3]) in MEASURE_TOKENS
        ):
            add_count(int(a), [i])

    # 5) Patrones tipo 12 GR X 6 X 60 UND
    if measure_idx >= 0:
        i = measure_idx + 2
        while i < len(tokens) - 1:
            if _token_word(tokens[i]) != "X":
                i += 1
                continue

            n = _token_numero(tokens[i + 1])
            if n is None or not float(n).is_integer():
                i += 1
                continue

            nxt = _token_word(tokens[i + 2]) if i + 2 < len(tokens) else ""
            if nxt in MEASURE_TOKENS or nxt in DIMENSION_TOKENS:
                break

            add_count(int(n), [i + 1])
            i += 2

    # 6) Patrones tipo 12 X 10 / 9 X 40 / 6 X 5 X 10
    for i in range(len(tokens) - 2):
        a = _token_numero(tokens[i])
        b = _token_numero(tokens[i + 2])

        if (
            a is None
            or b is None
            or not float(a).is_integer()
            or not float(b).is_integer()
            or _token_word(tokens[i + 1]) != "X"
        ):
            continue

        nxt = _token_word(tokens[i + 3]) if i + 3 < len(tokens) else ""
        if nxt in MEASURE_TOKENS or nxt in DIMENSION_TOKENS:
            continue

        cerca_del_final = i + 3 >= len(tokens)
        sigue_pack = nxt in PACK_TOKENS or nxt in {"EX"}
        if cerca_del_final or sigue_pack:
            add_count(int(a), [i])
            add_count(int(b), [i + 2])

    return counts, usados


def _limpiar_token_para_base(token: str) -> str:
    t = token.strip()
    if not t:
        return ""

    # Remover prefijos tipo 029-, 041-, 002-, etc
    t = re.sub(r"^\d{2,4}-", "", t)

    # Si queda vacío, descartar
    return t.strip()


def extraer_atributos_producto(texto: str) -> Dict[str, object]:
    texto_limpio = normalizar_texto(texto)
    tokens = _tokenizar_extraccion(texto)

    if not tokens:
        return {
            "Producto_limpio": "",
            "Producto_base_norm": "",
            "FactorConversion": 1.0,
            "ContenidoUnidad": 0.0,
            "ContenidoTotal": 0.0,
            "TipoContenido": "NONE",
            "PesoExtraidoKg": 0.0,
        }

    contenido_unidad, tipo_contenido, measure_idx = _extraer_medida(tokens)
    factores, usados = _extraer_factores(tokens, measure_idx)

    factor_conversion = 1.0
    for n in factores:
        factor_conversion *= float(n)

    contenido_total = contenido_unidad * factor_conversion if contenido_unidad > 0 else 0.0
    peso_extraido_kg = (contenido_unidad / 1000.0) if tipo_contenido == "MASS" else 0.0

    tokens_base: List[str] = []
    for i, tok in enumerate(tokens):
        t = _token_word(tok)
        if not t:
            continue

        if i in usados:
            continue

        if measure_idx >= 0 and i in {measure_idx, measure_idx + 1}:
            continue

        t = _limpiar_token_para_base(t)
        if not t:
            continue

        if t in PACK_TOKENS or t in MEASURE_TOKENS or t in {"X"}:
            continue

        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            prev_t = _token_word(tokens[i - 1]) if i > 0 else ""
            next_t = _token_word(tokens[i + 1]) if i + 1 < len(tokens) else ""

            es_numero_marca = (
                t.isdigit()
                and int(float(t)) <= 10
                and prev_t not in PACK_TOKENS
                and next_t not in PACK_TOKENS
                and prev_t not in MEASURE_TOKENS
                and next_t not in MEASURE_TOKENS
                and prev_t != "X"
                and next_t != "X"
            )

            if es_numero_marca:
                tokens_base.append(t)

            continue

        if re.fullmatch(r"\d+-\d+", t):
            continue

        if t in NOISE_TOKENS:
            continue

        if re.fullmatch(r"RN\d+", t):
            continue

        if re.fullmatch(r"\d{4}-\d{2}", t):
            continue

        tokens_base.append(t)

    producto_base_norm = " ".join(tokens_base).strip()
    if not producto_base_norm:
        producto_base_norm = texto_limpio

    return {
        "Producto_limpio": texto_limpio,
        "Producto_base_norm": producto_base_norm,
        "FactorConversion": float(factor_conversion),
        "ContenidoUnidad": float(contenido_unidad),
        "ContenidoTotal": float(contenido_total),
        "TipoContenido": tipo_contenido,
        "PesoExtraidoKg": float(peso_extraido_kg),
    }


def construir_texto_modelo(
    producto_base_norm: str,
    factor_conversion: float,
    contenido_unidad: float,
    tipo_contenido: str,
) -> str:
    partes = [producto_base_norm.strip()]

    if factor_conversion and factor_conversion > 1:
        partes.append(f"FC_{int(round(factor_conversion))}")

    if contenido_unidad and contenido_unidad > 0:
        partes.append(f"CONT_{int(round(contenido_unidad))}")

    if tipo_contenido and tipo_contenido != "NONE":
        partes.append(f"TIPO_{tipo_contenido}")

    texto = " ".join(p for p in partes if p).strip()
    return re.sub(r"\s+", " ", texto)


def extraer_peso_desde_texto(texto: str) -> float:
    attrs = extraer_atributos_producto(texto)
    return float(attrs["PesoExtraidoKg"])


def log_seguro(x: float) -> float:
    x = 0.0 if pd.isna(x) else float(x)
    return float(math.log1p(max(x, 0.0)))