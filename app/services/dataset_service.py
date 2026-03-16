from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import pyodbc

from app.core.config import CONFIG_SEDES, get_settings


SQL_MAESTRO_REX = """
WITH UltimoCosto AS
(
    SELECT
        ducppm.CodProveedor,
        ducppm.CodCategoria,
        ducppm.Categoria,
        ducppm.CodMarca,
        ducppm.Marca,
        ducppm.CodProducto,
        ducppm.Productos AS Producto,
        ducppm.AñoEmision,
        ducppm.MesEmision,
        ducppm.CostoUnitarioUltimo,
        ROW_NUMBER() OVER (
            PARTITION BY ducppm.CodProducto
            ORDER BY ducppm.AñoEmision DESC, ducppm.MesEmision DESC
        ) AS rn
    FROM dev_UltimoCostoProductoPorMes (NOLOCK) ducppm
),
MarcasConfiguradas AS (
    SELECT
        DISTINCT
        (a.carticulos_id) AS CodProducto,
        RTRIM(CASE
            WHEN a.cproveedor_id = '00000002' THEN c1.cdescripcion_cat
            WHEN a.cproveedor_id = '00000041' THEN c2.cdescripcion_cat
            WHEN a.cproveedor_id = '00000048' THEN c2.cdescripcion_cat
            WHEN a.cproveedor_id = '00000001' THEN c1.cdescripcion_cat
            WHEN a.cproveedor_id = '00000025' THEN c4.cdescripcion_cat
            WHEN a.cproveedor_id = '00000029' THEN c4.cdescripcion_cat
            WHEN a.cproveedor_id = '00000050' THEN c2.cdescripcion_cat
            ELSE c1.cdescripcion_cat
        END) AS Marca
    FROM _articulos a
    LEFT JOIN _categoria_1 c1 ON c1.ccodigo_categoria = a.ccategoria_1
    LEFT JOIN _categoria_2 c2 ON c2.ccodigo_categoria = a.ccategoria_2
    LEFT JOIN _categoria_4 c4 ON c4.ccodigo_categoria = a.ccategoria_4
),
CategoriasConfiguradas AS (
    SELECT
        DISTINCT
        a.carticulos_id AS CodProducto,
        RTRIM(CASE
            WHEN a.cproveedor_id = '00000002' THEN c2.cdescripcion_cat
            WHEN a.cproveedor_id = '00000041' THEN c1.cdescripcion_cat
            WHEN a.cproveedor_id = '00000048' THEN c1.cdescripcion_cat
            WHEN a.cproveedor_id = '00000001' THEN c2.cdescripcion_cat
            WHEN a.cproveedor_id = '00000025' THEN c2.cdescripcion_cat
            WHEN a.cproveedor_id = '00000029' THEN c2.cdescripcion_cat
            WHEN a.cproveedor_id = '00000050' THEN c1.cdescripcion_cat
            ELSE c2.cdescripcion_cat
        END) AS Categoria
    FROM _articulos a
    LEFT JOIN _categoria_1 c1 ON c1.ccodigo_categoria = a.ccategoria_1
    LEFT JOIN _categoria_2 c2 ON c2.ccodigo_categoria = a.ccategoria_2
    LEFT JOIN _categoria_4 c4 ON c4.ccodigo_categoria = a.ccategoria_4
)
SELECT
    RTRIM(p.cproveedor_ruc) AS RucProveedor,
    cc.Categoria,
    mc.Marca,
    RTRIM(a.carticulos_id) AS CodProducto,
    RTRIM(a.carticulos_codigo_equipo) AS CodProducto2,
    RTRIM(a.carticulos_nro_parte) AS CodProducto3,
    RTRIM(a.carticulos_nombre) AS Producto,
    RTRIM(um.cunidad_de_medida_nombre) AS UnidaMedidaCompra,
    a.nfactor_a_venta AS FactorVenta,
    a.narticulos_factor_conversion AS FactorConversion,
    ROUND(a.narticulos_peso / a.narticulos_factor_conversion, 2) AS PesoUnitario,
    a.narticulos_peso AS PesoCaja,
    uc.CostoUnitarioUltimo * a.narticulos_factor_conversion AS CostoCaja
FROM [_articulos] a
INNER JOIN _proveedor p ON p.cproveedor_id = a.cproveedor_id
INNER JOIN _unidad_de_medida um ON um.cunidad_de_medida_id = a.cunidad_de_medida_compra
LEFT JOIN UltimoCosto uc ON uc.CodProducto = a.carticulos_id AND uc.rn = 1
LEFT JOIN MarcasConfiguradas mc ON mc.CodProducto = a.carticulos_id
LEFT JOIN CategoriasConfiguradas cc ON cc.CodProducto = a.carticulos_id
WHERE cproveedor_ruc <> ''
ORDER BY RucProveedor ASC;
"""

SQL_FACTURAS_NUEVAS_REX = """
WITH UltimoCosto AS (
    SELECT c.EmisorRuc, COALESCE(ci.CodProducto, ci.CodProdGS1) AS CodProducto,
           ci.Descripcion AS Producto, ci.DesUnidadMedida AS UnidaMedidaCompra,
           ci.MtoPrecioUnitario AS CostoCaja,
           ROW_NUMBER() OVER (PARTITION BY COALESCE(ci.CodProducto, ci.CodProdGS1) ORDER BY c.FechaEmision DESC, c.CarSunat DESC) AS rn
    FROM Cpe c INNER JOIN CpeItems ci ON ci.CarSunat = c.CarSunat
    WHERE c.TipoDoc = '01' AND LEN(COALESCE(ci.CodProducto, ci.CodProdGS1)) > 2
      AND c.EmisorRuc IN (SELECT cproveedor_ruc FROM _proveedor)
)
SELECT EmisorRuc AS RucProveedor, CodProducto, Producto, UnidaMedidaCompra, CostoCaja FROM UltimoCosto WHERE rn = 1;
"""


SQL_MAESTRO_PORTALES = """
WITH UltimoCosto AS (
    SELECT imd.ProductoId, im.FechaEmision, imd.PrecioUnitario AS CostoUnitario,
           ROW_NUMBER() OVER (PARTITION BY imd.ProductoId ORDER BY im.FechaEmision DESC) AS rn
    FROM IngresoMercaderiaDetalle imd
    INNER JOIN IngresoMercaderia im ON im.Id = imd.IngresoMercaderiaId
    WHERE im.EstadoRegistro = 1
),
MarcasConfiguradas AS (
    SELECT p.Id AS CodProducto, m.Descripcion AS Marca FROM Producto p LEFT JOIN Marca m ON m.Id = p.MarcaId
),
CategoriasConfiguradas AS (
    SELECT p.Id AS CodProducto, f.Descripcion AS Categoria FROM Producto p LEFT JOIN Familia f ON f.FamiliaId = p.FamiliaId
)
SELECT DISTINCT
    RTRIM(pe.NroDocumento) AS RucProveedor, cc.Categoria, mc.Marca,
    RTRIM(p.Id) AS CodProducto, RTRIM(p.CodigoInterno) CodProducto2,
    RTRIM(p.Serie) AS CodProducto3, RTRIM(p.DescripcionLarga) AS Producto,
    RTRIM(um.Descripcion) AS UnidadMedidaCompra, pd.FactorConversion AS FactorVenta,
    pd.FactorConversion AS FactorConversion,
    CASE WHEN pd.FactorConversion <> 0 THEN pd.Peso / pd.FactorConversion ELSE NULL END AS PesoUnitario,
    pd.Peso AS PesoCaja,
    CASE WHEN uc.CostoUnitario IS NOT NULL THEN uc.CostoUnitario * pd.FactorConversion ELSE NULL END AS CostoCaja
FROM Producto p
INNER JOIN ProductoDetalle pd ON pd.ProductoId = p.Id
INNER JOIN UnidadMedida um ON um.Id = pd.UnidadMedidaId
LEFT JOIN CarteraProducto cp ON cp.ProductoId = p.Id
LEFT JOIN Proveedor pr ON pr.Id = cp.ProveedorId
LEFT JOIN Persona pe ON pe.Id = pr.PersonaId
LEFT JOIN UltimoCosto uc ON uc.ProductoId = p.Id AND uc.rn = 1
LEFT JOIN MarcasConfiguradas mc ON mc.CodProducto = p.Id
LEFT JOIN CategoriasConfiguradas cc ON cc.CodProducto = p.Id
WHERE pe.NroDocumento IS NOT NULL AND pe.NroDocumento <> ''
ORDER BY RucProveedor ASC, CodProducto ASC;
"""

SQL_FACTURAS_NUEVAS_PORTALES = """
WITH UltimoCosto AS (
    SELECT 
        c.EmisorRuc, 
        COALESCE(ci.CodProducto, ci.CodProdGS1) AS CodProducto, 
        ci.Descripcion AS Producto, 
        ci.DesUnidadMedida AS UnidaMedidaCompra,
        ci.MtoPrecioUnitario AS CostoCaja,
        ROW_NUMBER() OVER (
            PARTITION BY COALESCE(ci.CodProducto, ci.CodProdGS1) 
            ORDER BY c.FechaEmision DESC, c.CarSunat DESC
        ) AS rn
    FROM Cpe c
    INNER JOIN CpeItems ci ON ci.CarSunat = c.CarSunat
    WHERE c.TipoDoc = '01' AND LEN(COALESCE(ci.CodProducto, ci.CodProdGS1)) > 2
    AND c.EmisorRuc IN (SELECT NroDocumento FROM Persona)
)
SELECT 
    EmisorRuc AS RucProveedor, 
    CodProducto, 
    Producto, 
    UnidaMedidaCompra, 
    CostoCaja
FROM UltimoCosto
WHERE rn = 1
ORDER BY RucProveedor ASC;
"""


QUERIES_MAP = {
    "PORTALES": {
        "maestro": SQL_MAESTRO_PORTALES,
        "facturas": SQL_FACTURAS_NUEVAS_PORTALES
    },
    "REX": {
        "maestro": SQL_MAESTRO_REX,
        "facturas": SQL_FACTURAS_NUEVAS_REX
    }
}


def _get_queries_for_sede(sede: str) -> dict:
    sede_upper = sede.upper()
    if "PORTALES" in sede_upper:
        return QUERIES_MAP["PORTALES"]
    return QUERIES_MAP["REX"]

def _build_connection_string(sede: str) -> str:
    sede_key = sede.upper().strip()
    if sede_key not in CONFIG_SEDES:
        raise ValueError(f"Sede '{sede}' no configurada.")

    cfg = CONFIG_SEDES[sede_key]
    driver = os.getenv("SQLSERVER_DRIVER", "ODBC Driver 17 for SQL Server")
    return (
        f"DRIVER={{{driver}}};SERVER={cfg.host},{cfg.port};DATABASE={cfg.database};"
        f"UID={cfg.user};PWD={cfg.password};Encrypt=no;TrustServerCertificate=yes;"
    )

def _read_sql(sql: str, sede: str) -> pd.DataFrame:
    conn_str = _build_connection_string(sede)
    with pyodbc.connect(conn_str) as conn:
        return pd.read_sql(sql, conn)

def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df

def generar_datasets_raw(sede: str, overwrite: bool = True) -> dict:
    settings = get_settings()
    queries = _get_queries_for_sede(sede)

    print(f"Generando datasets para {sede}...")

    maestro_df = _normalize_text_columns(_read_sql(queries["maestro"], sede))
    facturas_df = _normalize_text_columns(_read_sql(queries["facturas"], sede))

    maestro_path = settings.raw_data_dir / "maestro.csv"
    facturas_path = settings.raw_data_dir / "productos_facturas.csv"

    maestro_path.parent.mkdir(parents=True, exist_ok=True)
    maestro_df.to_csv(maestro_path, index=False, encoding="utf-8-sig")
    facturas_df.to_csv(facturas_path, index=False, encoding="utf-8-sig")

    return {
        "sede": sede.upper(),
        "maestro": {
            "nombre": "maestro.csv",
            "filas": len(maestro_df), 
            "ruta": str(maestro_path)
        },
        "productos_facturas": {
            "nombre": "productos_facturas.csv",
            "filas": len(facturas_df), 
            "ruta": str(facturas_path)
        }
    }