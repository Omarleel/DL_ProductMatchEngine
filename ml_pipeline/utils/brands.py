from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

MARCAS_COMUNES  = [
    "BUBBALOO", "CHICLETS", "CLORETS", "CLUB SOCIAL", "FIELD", "HALLS", "RITZ", "TRIDENT",
    "ARUBA", "BELLA HOLANDESA", "BONLE", "CARTAVIO", "CASA GRANDE", "CHICOLAC", "COMPLETE",
    "DOLCCI", "DUMMY", "ECOLAT", "GLORIA", "LA FLORENCIA", "LA MESA", "LA PRADERA", "LEAF TEA",
    "MILKITO", "PRO", "PURA VIDA", "SAN MATEO", "SHAKE", "SOY VIDA", "TAMPICO", "YOFRESH",
    "YOMOST", "ZEROLACTO", "BARBIE", "CARS", "CLASICA TAN", "CLASICA URBANA", "HAPPY 1",
    "MAIS LIKE", "MAIS TRANCE", "PRINCESA", "SALOME", "SPIDERMAN", "URBANA", "BEIERSDORF",
    "COLGATE", "CREST", "DERMEX", "EFILA", "FAPE", "GENOMA LAB", "GOOD BRANDS", "GSK",
    "HERSIL", "INCASUR", "INTRADEVCO", "LEP", "MARUCHAN", "NINET", "NIVEA", "P&G", "PANASONIC",
    "RECKITT", "NOEL", "TMLUC", "PARACAS", "PETALO", "SEDITA", "ALACENA", "ALIANZA", "AMARAS",
    "AMOR", "ANGEL", "AVAL", "BLANCA FLOR", "BOLIVAR", "CAPRI", "CASINO", "COCINERO", "DENTO",
    "DIA", "DON VITTORIO", "ESPIGA DE ORO", "GEOMEN", "JUMBO", "LA FAVORITA", "LAVAGGI",
    "MARSELLA", "NICOLINI", "NORCHEFF", "OPAL", "PATITO", "PREMIO", "PRIMOR", "SAPOLIO",
    "SAYON", "SELLO DE ORO", "TROME", "UMSHA", "VICTORIA", "MIMASKOT", "3 OSITOS", "COSTA",
    "CAÑONAZO", "CAREZZA", "MOLITALIA", "MARCO POLO", "POMAROLA", "FANNY", "NUTRICAN",
    "PASCUALINO", "TODINNO", "VIZZIO", "CARICIA", "MONCLER", "ÑA PANCHA", "DON MANOLO",
    "FORTUNA", "MANPAN", "PALMEROLA", "POPEYE", "SPA", "TONDERO", "ACE", "ALWAYS", "ARIEL",
    "AYUDIN", "CLEARBLUE", "DOWNY", "GILLETTE", "HEPABIONTA", "HERBAL ESSENCES", "H&S",
    "OLD SPICE", "ORAL B", "PAMPERS", "PANTENE", "SECRET", "VICK", "BANDIDO", "CANBO",
    "FRESHCAN", "MICHICAT", "REX", "RICOCAN", "RICOCAT", "RICOCRACK", "SUPERCAN", "SUPERCAT",
    "THOR", "YAMIS DOG", "VUSE", "INKA KOLA", "CUSQUEÑA", "CRISTAL", "PILSEN CALLAO", "SUBLIME",
    "DONOFRIO", "TRIANGULO", "MOROCHAS", "CHARADA", "PIQUEO SNAX", "LAY'S", "DORITOS", "CHEETOS", 
    "CUATES", "CIELO", "PULP", "AJE", "VOLT", "SPORADE", "SABIDIA", "CUMANÁ", "WINTERS", "GN", 
    "BOLTS", "CUA CUA", "KIRKLAND", "MAGGI", "NESTLÉ", "NESCAFÉ", "MILO", "IDEAL", "LA LECHERA", "KIRITOS",
    "COCA COLA", "FRUGOS", "SPRITE", "FANTA", "POWERADE", "AQUAFINA", "PEPSI", "SEVEN UP",
    "GATORADE", "CONCORDIA", "GUARANA", "SAN CARLOS", "CORONA", "BRAHMA", "PILSEN TRUJILLO",
    "AREQUIPEÑA", "PILSEN POLAR", "FREE TEA", "BIO AMARANTE", "KR", "ORO", "BIG COLA",
    "CHITOS", "CHEESE TRIS", "TORTEES", "MANI MOTO", "KARINTO", "FRITO LAY",
    "HUGGIES", "KOTEX", "SUAVE", "PLENIT", "SCOTT", "ELITE", "NOBLE", "BABYSEC",
    "LADYSOFT", "POISE", "TENA", "REDOXON", "APRILIS", "UMSHA", "BAMBINO",
    "ANITA", "COSTEÑO", "PAISANA", "VALLE NORTE", "HOJALMAR", "GRETEL", "CHINCHIN",
    "GRANJA AZUL", "LA CALERA", "HUACARIZ", "VIGOR", "DANLAC", "TAYTA",
    "TODINNO", "BAUDUCCO", "MOTTA", "BLANCA FLOR", "SAYON", "DORINA", "MANTEY",
    "LA DANESA", "MANTY", "STEVIA", "SUCRALOSA", "GLADE", "POETT", "CLOROX",
    "PINO", "ZORRITO", "PATITO", "SILJET", "RAID", "OFF", "CERINI", "VAPO",
    "SAN JORGE", "GN", "VICTORIA", "MC CORMICK", "BADIA", "KIKKOMAN", "HEINZ", "HELLMANNS", 
    "KRAFT", "PHILADELPHIA", "MT. OLIVE", "BARILLA", "AGNESI", "DE CECCO", "DON ITALO",
    "COMPASS", "CAMPBELLS", "QUAKER", "KELLOGGS", "NESTUM", "CERELAC", "A-1", "SARDIMAR",
    "REAL", "CAMPOMAR", "FLORIDA", "HERENCIA", "BON AMI", "ALBORADA", "EL OLIVAR",
    "OTTO KUNZ", "LA SEGURIA", "SAN FERNANDO", "REDONDOS", "BRAEDT", "SUIZA", "ZURICH",
    "CASERIO", "LAIVE", "CERDEÑA", "STEFFANO", "RAZZETO", "EL POZO",
    "PRINGLES", "STAX", "TORTREES", "PEANUTS", "NUTS", "NUCITA", "TIC TAC", "M&M", 
    "SNICKERS", "MILKY WAY", "KINDER", "FERRERO ROCHER", "HERSHEYS", "KIT KAT",
    "TOBLERONE", "LOTUS", "OREO", "CHIPS AHOY", "MAMBA", "SKITTLES", "MENTOS",
    "ADEZ", "VIVESOY", "SILK", "ALMOND BREEZE", "NATURES HEART", "MOLITALIA", "KASERI",
    "HEINEKEN", "STELLA ARTOIS", "BUDWEISER", "MILLER", "HOEGAARDEN", "CORONA EXTRA",
    "SANTIAGO QUEIROLO", "TACAMA", "TABERNERO", "INTIPALKA", "OCUCAJE", "VINA VIEJA",
    "JOHNNIE WALKER", "OLD PARR", "CHIVAS REGAL", "JACK DANIELS", "BALLANTINES",
    "SMIRNOFF", "ABSOLUT", "SKYY", "JOSE CUERVO", "FOUR LOKO", "MIKE'S",
    "DOVE", "REXONA", "AXE", "LUX", "CAMAY", "LISTERINE", "JOHNSONS", "NEUTROGENA",
]

def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_brand_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).upper().strip()
    text = _strip_accents(text)
    text = text.replace("&", " ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for v in values:
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


@lru_cache(maxsize=1)
def normalized_brand_keys() -> list[str]:
    vals = [normalize_brand_text(x) for x in MARCAS_COMUNES if str(x).strip()]
    return _dedupe_keep_order(vals)


@lru_cache(maxsize=1)
def compiled_brand_patterns() -> list[tuple[str, re.Pattern]]:
    brands = normalized_brand_keys()
    # Más largas primero para capturar "HERBAL ESSENCES" antes que "ESSENCES"
    brands = sorted(brands, key=lambda s: (-len(s.split()), -len(s), s))
    patterns: list[tuple[str, re.Pattern]] = []
    for brand in brands:
        pat = re.compile(rf"(?<![A-Z0-9]){re.escape(brand)}(?![A-Z0-9])")
        patterns.append((brand, pat))
    return patterns


def extract_brand_hits(text: str) -> list[str]:
    txt = normalize_brand_text(text)
    if not txt:
        return []

    hits: list[str] = []
    for brand, pat in compiled_brand_patterns():
        if pat.search(txt):
            hits.append(brand)

    return _dedupe_keep_order(hits)


def extract_primary_brand(text: str) -> str:
    hits = extract_brand_hits(text)
    return hits[0] if hits else ""


def brand_set(text: str) -> set[str]:
    return set(extract_brand_hits(text))