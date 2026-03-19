import pandas as pd

from .limpieza import (
    construir_texto_modelo,
    extraer_atributos_producto,
    log_seguro,
    normalizar_codigo,
    normalizar_unidad,
    renombrar_columnas_equivalentes,
    validar_columnas,
)


EQUIVALENCIAS_MAESTRO = {
    "RucProveedor": ["rucproveedor", "ruc_proveedor", "ruc"],
    "CodProducto": ["codigo_producto", "codigoproducto", "codigo", "codproducto"],
    "CodProducto2": ["codigo_producto2", "codigoproducto2", "codproducto2"],
    "CodProducto3": ["codigo_producto3", "codigoproducto3", "codproducto3"],
    "Producto": ["producto", "descripcion", "descripción"],
    "UnidadMedidaCompra": ["unidadmedidacompra", "unidamedidacompra", "unidad_compra", "unidad"],
    "CostoCaja": ["costocaja", "costo_caja", "costo"],
}

EQUIVALENCIAS_FACTURAS = {
    "RucProveedor": ["rucproveedor", "ruc_proveedor", "ruc"],
    "CodProducto": ["codigo_producto", "codigoproducto", "codigo", "codproducto"],
    "Producto": ["producto", "descripcion", "descripción"],
    "UnidadMedidaCompra": [
        "unidadmedidacompra",
        "unidamedidacompra",
        "unidad_compra",
        "unidad",
        "unidadmedida",
    ],
    "CostoCaja": ["costocaja", "costo_caja", "costo"],
}


def _aplicar_atributos_producto(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    attrs = df["Producto"].fillna("").astype(str).map(extraer_atributos_producto)
    attrs_df = pd.DataFrame(list(attrs))

    columnas_attrs = [
        "Producto_limpio",
        "Producto_base_norm",
        "FactorConversion",
        "ContenidoUnidad",
        "ContenidoTotal",
        "TipoContenido",
        "PesoExtraidoKg",
    ]

    # Garantizar columnas aunque alguna extracción venga incompleta
    defaults = {
        "Producto_limpio": "",
        "Producto_base_norm": "",
        "FactorConversion": 1.0,
        "ContenidoUnidad": 0.0,
        "ContenidoTotal": 0.0,
        "TipoContenido": "NONE",
        "PesoExtraidoKg": 0.0,
    }

    for c in columnas_attrs:
        if c not in attrs_df.columns:
            attrs_df[c] = defaults[c]

    attrs_df = attrs_df[columnas_attrs]

    # Evitar duplicados si el DF ya venía con esas columnas
    df = df.drop(columns=[c for c in columnas_attrs if c in df.columns], errors="ignore")

    df = pd.concat(
        [df.reset_index(drop=True), attrs_df.reset_index(drop=True)],
        axis=1,
    )

    df["TipoContenido"] = df["TipoContenido"].fillna("NONE").astype(str)
    df["FactorConversion"] = pd.to_numeric(df["FactorConversion"], errors="coerce").fillna(1.0)
    df["ContenidoUnidad"] = pd.to_numeric(df["ContenidoUnidad"], errors="coerce").fillna(0.0)
    df["ContenidoTotal"] = pd.to_numeric(df["ContenidoTotal"], errors="coerce").fillna(0.0)
    df["PesoExtraidoKg"] = pd.to_numeric(df["PesoExtraidoKg"], errors="coerce").fillna(0.0)

    df["Factor_log"] = df["FactorConversion"].map(log_seguro)
    df["ContenidoUnidad_log"] = df["ContenidoUnidad"].map(log_seguro)
    df["ContenidoTotal_log"] = df["ContenidoTotal"].map(log_seguro)

    df["Producto_norm"] = df.apply(
        lambda r: construir_texto_modelo(
            producto_base_norm=r["Producto_base_norm"],
            factor_conversion=r["FactorConversion"],
            contenido_unidad=r["ContenidoUnidad"],
            tipo_contenido=r["TipoContenido"],
        ),
        axis=1,
    )

    return df


def preparar_maestro(maestro: pd.DataFrame) -> pd.DataFrame:
    df = renombrar_columnas_equivalentes(maestro, EQUIVALENCIAS_MAESTRO)

    validar_columnas(
        df,
        ["RucProveedor", "CodProducto", "Producto", "UnidadMedidaCompra"],
        "maestro",
    )

    for c in ["CodProducto", "CodProducto2", "CodProducto3"]:
        if c not in df.columns:
            df[c] = ""

        df[c] = (
            df[c]
            .fillna("")
            .astype(str)
            .replace("nan", "")
            .map(normalizar_codigo)
        )

    if "CostoCaja" not in df.columns:
        df["CostoCaja"] = 0.0

    if "PesoUnitario" not in df.columns:
        df["PesoUnitario"] = 0.0

    df["PesoUnitario"] = pd.to_numeric(df["PesoUnitario"], errors="coerce").fillna(0.0)
    df["Unidad_norm"] = df["UnidadMedidaCompra"].fillna("").astype(str).map(normalizar_unidad)
    df["CostoCaja"] = pd.to_numeric(df["CostoCaja"], errors="coerce").fillna(0.0)
    df["Costo_log"] = df["CostoCaja"].map(log_seguro)

    df = _aplicar_atributos_producto(df)

    # Si el maestro no trae peso, usar lo extraído del texto.
    df["PesoUnitario"] = df["PesoUnitario"].where(df["PesoUnitario"] > 0, df["PesoExtraidoKg"])
    df["PesoTotalKg"] = df["PesoUnitario"] * df["FactorConversion"]
    df["PesoTotalKg_log"] = df["PesoTotalKg"].map(log_seguro)

    return df


def preparar_facturas(facturas: pd.DataFrame) -> pd.DataFrame:
    df = renombrar_columnas_equivalentes(facturas, EQUIVALENCIAS_FACTURAS)

    validar_columnas(
        df,
        ["RucProveedor", "CodProducto", "Producto", "UnidadMedidaCompra"],
        "facturas",
    )

    if "CostoCaja" not in df.columns:
        df["CostoCaja"] = 0.0

    if "PesoUnitario" not in df.columns:
        df["PesoUnitario"] = 0.0

    df["CodProducto"] = (
        df["CodProducto"]
        .fillna("")
        .astype(str)
        .replace("nan", "")
        .map(normalizar_codigo)
    )

    df["Unidad_norm"] = df["UnidadMedidaCompra"].fillna("").astype(str).map(normalizar_unidad)
    df["CostoCaja"] = pd.to_numeric(df["CostoCaja"], errors="coerce").fillna(0.0)
    df["Costo_log"] = df["CostoCaja"].map(log_seguro)
    df["PesoUnitario"] = pd.to_numeric(df["PesoUnitario"], errors="coerce").fillna(0.0)

    df = _aplicar_atributos_producto(df)

    df["PesoUnitario"] = df["PesoUnitario"].where(df["PesoUnitario"] > 0, df["PesoExtraidoKg"])
    df["PesoTotalKg"] = df["PesoUnitario"] * df["FactorConversion"]
    df["PesoTotalKg_log"] = df["PesoTotalKg"].map(log_seguro)

    return df