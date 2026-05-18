from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import pyodbc

from app.core.config import CONFIG_SEDES, get_settings
from app.services.maestro_service import limpiar_cache_maestro
from app.services.tenant_service import get_tenant_raw_data_dir, normalizar_tenant
from ml_pipeline.utils.limpieza import normalizar_codigo, normalizar_unidad


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
    RTRIM(um.cunidad_de_medida_nombre) AS UnidadMedidaCompra,
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
           ci.Descripcion AS Producto, ci.DesUnidadMedida AS UnidadMedidaCompra,
           ci.MtoPrecioUnitario AS CostoCaja,
           ROW_NUMBER() OVER (PARTITION BY COALESCE(ci.CodProducto, ci.CodProdGS1) ORDER BY c.FechaEmision DESC, c.CarSunat DESC) AS rn
    FROM Cpe c INNER JOIN CpeItems ci ON ci.CarSunat = c.CarSunat
    WHERE c.TipoDoc = '01' AND LEN(COALESCE(ci.CodProducto, ci.CodProdGS1)) > 2
      AND c.EmisorRuc IN (SELECT cproveedor_ruc FROM _proveedor)
)
SELECT EmisorRuc AS RucProveedor, CodProducto, Producto, UnidadMedidaCompra, CostoCaja FROM UltimoCosto WHERE rn = 1;
"""


SQL_AUTOHOMOLOGACION_REX = """
WITH CpeItemsNumerados AS (
    SELECT
        ci.CarSunat,
        COALESCE(ci.CodProducto, ci.CodProdGS1) AS CodProducto,
        ci.Descripcion,
        ci.Cantidad,
        ci.CodUnidadMedida,
        ci.Total,
        ROW_NUMBER() OVER (
            PARTITION BY ci.CarSunat
            ORDER BY Id
        ) AS NumItem
    FROM CpeItems ci (NOLOCK) 
    INNER JOIN Cpe c (NOLOCK) ON ci.CarSunat = c.CarSunat AND c.TipoDoc = '01'
),
MatchesPaginados AS (
    SELECT
        RTRIM(p.cproveedor_ruc) AS RucProveedor,
        RTRIM(a.carticulos_id) AS CodProductoMaestro,
        a.carticulos_nombre AS ProductoMaestro,
        cin.CodProducto AS CodProductoCpe,
        cin.Descripcion AS ProductoCpe,
        ROW_NUMBER() OVER (
            PARTITION BY RTRIM(p.cproveedor_ruc), RTRIM(a.carticulos_id), cin.CodProducto
            ORDER BY m.dmovimiento_fechahora DESC, m.cmovimiento_id DESC
        ) AS Fila
    FROM [_movimiento] m (NOLOCK)
    INNER JOIN [_movimiento_detalle] md (NOLOCK) ON md.cmovimiento_id = m.cmovimiento_id
    INNER JOIN [_proveedor] p ON p.cproveedor_id = m.cproveedor_id
    INNER JOIN [_articulos] a ON a.carticulos_id = md.carticulos_id
    INNER JOIN CpeItemsNumerados cin
        ON CONCAT(cin.CarSunat, RIGHT('000' + CAST(cin.NumItem AS VARCHAR(3)), 3)) = CONCAT(
            LTRIM(RTRIM(p.cproveedor_ruc)),
            '01',
            SUBSTRING(LTRIM(RTRIM(m.cmovimiento_nro_sunat)), 1, 4),
            RIGHT('0000000000' + SUBSTRING(LTRIM(RTRIM(m.cmovimiento_nro_sunat)), 5, 20), 10),
            RIGHT('000' + LTRIM(RTRIM(md.cmovimiento_detalle_item)), 3)
        )
    WHERE m.dmovimiento_fechahora >= DATEADD(YEAR, -2, CAST(GETDATE() AS DATE))
      AND m.calmacenes_origen_id = '001'
      AND m.cmovimiento_estado = 'Ad'
      AND m.cmovimiento_tipo = 'I'
      AND LTRIM(RTRIM(m.cmovimiento_nro_sunat)) LIKE '%[0-9]'
      AND ABS(cin.Total - md.nmovimiento_detalle_valor_venta_soles) < 1
)
SELECT
    RucProveedor,
    CodProductoMaestro,
    ProductoMaestro,
    CodProductoCpe,
    ProductoCpe
FROM MatchesPaginados
WHERE Fila = 1
ORDER BY CodProductoMaestro ASC;
"""


SQL_CONVERSION_UNIDADES_REX = """
WITH CpeItemsNumerados AS (
    SELECT 
        ci.CarSunat, 
        COALESCE(ci.CodProducto, ci.CodProdGS1) AS CodProducto, 
        ci.Descripcion, 
        ci.Cantidad,
        ci.CodUnidadMedida,
        ci.Total,
        ROW_NUMBER() OVER (
            PARTITION BY ci.CarSunat 
            ORDER BY ci.Id
        ) AS NumItem
    FROM CpeItems ci
    INNER JOIN Cpe c ON ci.CarSunat = c.CarSunat AND c.TipoDoc = '01'
)
SELECT 
    CONCAT(
        LTRIM(RTRIM(p.cproveedor_ruc)),
        '01',
        SUBSTRING(LTRIM(RTRIM(m.cmovimiento_nro_sunat)), 1, 4),
        RIGHT('0000000000' + SUBSTRING(LTRIM(RTRIM(m.cmovimiento_nro_sunat)), 5, 20), 10),
        RIGHT('000' + LTRIM(RTRIM(md.cmovimiento_detalle_item)), 3)
    ) AS CarSunatItem,
    LTRIM(RTRIM(p.cproveedor_ruc)) AS RucProveedor,
    RTRIM(a.carticulos_id) AS CodProductoMaestro,
    a.carticulos_nombre ProductoMaestro,
    md.ncantidad_compra AS CantidadCompra,
    md.cunidad_compra AS UnidadMedidaCompra,
    cin.CodProducto AS CodProductoCpe,
    cin.Descripcion AS ProductoCpe,
    cin.Cantidad AS CantidadCpe,
    cin.CodUnidadMedida AS UnidadMedidaCpe
FROM [_movimiento] m 
INNER JOIN [_movimiento_detalle] md ON md.cmovimiento_id = m.cmovimiento_id
INNER JOIN [_proveedor] p ON p.cproveedor_id = m.cproveedor_id
INNER JOIN [_articulos] a ON a.carticulos_id = md.carticulos_id 
LEFT JOIN CpeItemsNumerados cin 
    ON CONCAT(cin.CarSunat, RIGHT('000' + CAST(cin.NumItem AS VARCHAR(3)), 3)) = CONCAT(
        LTRIM(RTRIM(p.cproveedor_ruc)), 
        '01',                            
        SUBSTRING(LTRIM(RTRIM(m.cmovimiento_nro_sunat)), 1, 4), 
        RIGHT('0000000000' + SUBSTRING(LTRIM(RTRIM(m.cmovimiento_nro_sunat)), 5, 20), 10), 
        RIGHT('000' + LTRIM(RTRIM(md.cmovimiento_detalle_item)), 3) 
    )
WHERE m.dmovimiento_fechahora >= DATEADD(YEAR, -1, CAST(GETDATE() AS DATE))
  AND m.calmacenes_origen_id = '001'
  AND m.cmovimiento_estado = 'Ad'
  AND m.cmovimiento_tipo = 'I'
  AND LTRIM(RTRIM(m.cmovimiento_nro_sunat)) LIKE '%[0-9]'
  AND ABS(cin.Total - md.nmovimiento_detalle_valor_venta_soles) < 1
ORDER BY CarSunatItem ASC;
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
        ci.DesUnidadMedida AS UnidadMedidaCompra,
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
    UnidadMedidaCompra, 
    CostoCaja
FROM UltimoCosto
WHERE rn = 1
ORDER BY RucProveedor ASC;
"""
SQL_AUTOHOMOLOGACION_PORTALES = """
WITH CpeItemsNumerados AS (
    SELECT
        ci.CarSunat,
        COALESCE(ci.CodProducto, ci.CodProdGS1) AS CodProducto,
        ci.Descripcion,
        ci.Cantidad,
        ci.CodUnidadMedida,
        ci.Total,
        ROW_NUMBER() OVER (
            PARTITION BY ci.CarSunat
            ORDER BY Id
        ) AS NumItem
    FROM CpeItems ci (NOLOCK)
    INNER JOIN Cpe c (NOLOCK) ON ci.CarSunat = c.CarSunat AND c.TipoDoc = '01'
),
DetallesErpNumerados AS (
    SELECT 
        dd.Id,
        dd.CabeceraDocumentoId,
        dd.ProductoId,
        dd.PrecioTotal,
        ROW_NUMBER() OVER (
            PARTITION BY dd.CabeceraDocumentoId 
            ORDER BY dd.Id
        ) AS NumItemErp
    FROM DetalleDocumento dd (NOLOCK)
    WHERE dd.EstadoRegistro = 1
),
MatchesPaginados AS (
    SELECT 
        RTRIM(p2.NroDocumento) AS RucProveedor,
        pr.Id AS CodProductoMaestro, 
        RTRIM(pr.DescripcionLarga) AS ProductoMaestro,
        cin.CodProducto AS CodProductoCpe,
        cin.Descripcion AS ProductoCpe,
        ROW_NUMBER() OVER (
            PARTITION BY RTRIM(p2.NroDocumento), pr.Id, cin.CodProducto
            ORDER BY cd.FechaEmision DESC, cd.Id DESC
        ) AS Fila
    FROM CabeceraDocumento cd (NOLOCK)
    INNER JOIN DetallesErpNumerados dd (NOLOCK) ON dd.CabeceraDocumentoId = cd.Id 
    INNER JOIN Producto pr (NOLOCK) ON pr.Id = dd.ProductoId 
    INNER JOIN Proveedor p (NOLOCK) ON p.Id = cd.ProveedorId 
    INNER JOIN Persona p2 (NOLOCK) ON p2.Id = p.PersonaId
    INNER JOIN CpeItemsNumerados cin
        ON CONCAT(cin.CarSunat, RIGHT('000' + CAST(cin.NumItem AS VARCHAR(3)), 3)) = CONCAT(
            LTRIM(RTRIM(p2.NroDocumento)),                     -- RUC (11 chars)
            '01',                                              -- Tipo Doc ('01' para Facturas)
            SUBSTRING(LTRIM(RTRIM(cd.Serie)), 1, 4),           -- Serie (4 chars)
            RIGHT('0000000000' + LTRIM(RTRIM(cd.Correlativo)), 10), -- Correlativo (10 chars)
            RIGHT('000' + CAST(dd.NumItemErp AS VARCHAR(3)), 3)     -- Ítem calculado (3 chars)
        )
    WHERE cd.TipoDocumentoRelacionadoId IN (1, 2, 5, 7) 
      AND cd.EstadoRegistro = 1 
      AND cd.FechaEmision >= DATEADD(YEAR, -2, CAST(GETDATE() AS DATE))
      AND ABS(cin.Total - dd.PrecioTotal) < 1 
)
SELECT 
    RucProveedor,
    CodProductoMaestro,
    ProductoMaestro,
    CodProductoCpe,
    ProductoCpe
FROM MatchesPaginados
WHERE Fila = 1
ORDER BY CodProductoMaestro ASC;
"""

SQL_CONVERSION_UNIDADES_PORTALES = """
WITH CpeItemsNumerados AS (
    SELECT 
        ci.CarSunat, 
        COALESCE(ci.CodProducto, ci.CodProdGS1) AS CodProducto, 
        ci.Descripcion, 
        ci.Cantidad,
        ci.CodUnidadMedida,
        ci.Total,
        ROW_NUMBER() OVER (
            PARTITION BY ci.CarSunat 
            ORDER BY ci.Id
        ) AS NumItem
    FROM CpeItems ci
    INNER JOIN Cpe c ON ci.CarSunat = c.CarSunat AND c.TipoDoc = '01'
),
DetallesErpNumerados AS (
    SELECT 
        dd.Id,
        dd.CabeceraDocumentoId,
        dd.ProductoId,
        dd.Cantidad,
        dd.UnidadMedidaId,
        dd.PrecioTotal,
        ROW_NUMBER() OVER (
            PARTITION BY dd.CabeceraDocumentoId 
            ORDER BY dd.Id
        ) AS NumItemErp
    FROM DetalleDocumento dd (NOLOCK)
    WHERE dd.EstadoRegistro = 1
)
SELECT 
     CONCAT(
        LTRIM(RTRIM(p2.NroDocumento)),                     -- RUC (11 chars)
        '01',                                              -- Tipo Doc ('01' para Facturas)
        SUBSTRING(LTRIM(RTRIM(cd.Serie)), 1, 4),           -- Serie (4 chars)
        RIGHT('0000000000' + LTRIM(RTRIM(cd.Correlativo)), 10), -- Correlativo (10 chars)
        RIGHT('000' + CAST(dd.NumItemErp AS VARCHAR(3)), 3)     -- Ítem calculado (3 chars)
    ) AS CarSunatItem,
    LTRIM(RTRIM(p2.NroDocumento)) AS RucProveedor,
    RTRIM(pr.Id) AS CodProductoMaestro,
    RTRIM(pr.DescripcionLarga) ProductoMaestro,
    dd.Cantidad AS CantidadCompra,
    RTRIM(um.Descripcion) AS UnidadMedidaCompra,
    cin.CodProducto AS CodProductoCpe,
    cin.Descripcion AS ProductoCpe,
    cin.Cantidad AS CantidadCpe,
    cin.CodUnidadMedida AS UnidadMedidaCpe
FROM CabeceraDocumento cd (NOLOCK)
INNER JOIN DetallesErpNumerados dd (NOLOCK) ON dd.CabeceraDocumentoId = cd.Id 
INNER JOIN Producto pr (NOLOCK) ON pr.Id = dd.ProductoId 
INNER JOIN Proveedor p (NOLOCK) ON p.Id = cd.ProveedorId 
INNER JOIN Persona p2 (NOLOCK) ON p2.Id = p.PersonaId
INNER JOIN UnidadMedida um (NOLOCK) ON um.Id = dd.UnidadMedidaId
INNER JOIN CpeItemsNumerados cin
ON CONCAT(cin.CarSunat, RIGHT('000' + CAST(cin.NumItem AS VARCHAR(3)), 3)) = CONCAT(
    LTRIM(RTRIM(p2.NroDocumento)),                     -- RUC (11 chars)
    '01',                                              -- Tipo Doc ('01' para Facturas)
    SUBSTRING(LTRIM(RTRIM(cd.Serie)), 1, 4),           -- Serie (4 chars)
    RIGHT('0000000000' + LTRIM(RTRIM(cd.Correlativo)), 10), -- Correlativo (10 chars)
    RIGHT('000' + CAST(dd.NumItemErp AS VARCHAR(3)), 3)     -- Ítem calculado (3 chars)
)
WHERE cd.TipoDocumentoRelacionadoId IN (1, 2, 5, 7) 
    AND cd.EstadoRegistro = 1 
    AND cd.FechaEmision >= DATEADD(YEAR, -1, CAST(GETDATE() AS DATE))
    AND ABS(cin.Total - dd.PrecioTotal) < 1 
ORDER BY CarSunatItem ASC;
"""

QUERIES_MAP = {
    "PORTALES": {
        "maestro": SQL_MAESTRO_PORTALES,
        "facturas": SQL_FACTURAS_NUEVAS_PORTALES,
        "auto_match": SQL_AUTOHOMOLOGACION_PORTALES,
        "conversion_unidades": SQL_CONVERSION_UNIDADES_PORTALES,
    },
    "REX": {
        "maestro": SQL_MAESTRO_REX,
        "facturas": SQL_FACTURAS_NUEVAS_REX,
        "auto_match": SQL_AUTOHOMOLOGACION_REX,
        "conversion_unidades": SQL_CONVERSION_UNIDADES_REX,
    },
}


def _get_queries_for_sede(sede: str) -> dict:
    sede_upper = sede.upper()
    if "PORTALES" in sede_upper:
        return QUERIES_MAP["PORTALES"]
    return QUERIES_MAP["REX"]


def _get_query_for_sede(sede: str, query_name: str) -> str:
    """
    Devuelve la consulta para una sede/tenant.

    Para hacer la autohomologación extensible por base de datos, se puede configurar
    un archivo SQL por ambiente con esta convención:

        SQL_<SEDE>_<QUERY_NAME>_QUERY_FILE

    Ejemplo:
        SQL_MI_TENANT_AUTOHOMOLOGACION_QUERY_FILE=/opt/sql/autohomologacion_mi_tenant.sql

    Si no existe archivo configurado, se usa el mapa interno QUERIES_MAP.
    """
    sede_key = sede.upper().strip()
    query_key = query_name.lower().strip()
    env_key = f"SQL_{sede_key}_{query_key.upper()}_QUERY_FILE"
    query_file = os.getenv(env_key)

    if query_file:
        path = Path(query_file)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo SQL configurado en {env_key}: {path}")
        return path.read_text(encoding="utf-8")

    queries = _get_queries_for_sede(sede)
    sql = queries.get(query_key)
    if not sql:
        raise ValueError(
            f"No hay consulta '{query_key}' configurada para la sede '{sede}'. "
            f"Configura {env_key} o agrega la consulta a QUERIES_MAP."
        )
    return sql

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


def _normalizar_ruc_conversion(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _modo_factor_conversion(group: pd.DataFrame) -> pd.Series:
    """
    Devuelve la moda del factor CantidadCompra/CantidadCpe para un grupo.

    Se redondea el factor antes de calcular la moda porque las cantidades pueden venir
    con decimales mínimos de SQL Server. En empates se elige el factor con más muestras,
    luego el de mayor confianza y finalmente el menor factor para evitar sobreestimar.
    """
    counts = (
        group.groupby("_factor_redondeado", dropna=False)
        .size()
        .reset_index(name="MuestrasModa")
        .sort_values(["MuestrasModa", "_factor_redondeado"], ascending=[False, True])
        .reset_index(drop=True)
    )

    factor = float(counts.iloc[0]["_factor_redondeado"])
    muestras_moda = int(counts.iloc[0]["MuestrasModa"])
    muestras_total = int(len(group))
    confianza = muestras_moda / muestras_total if muestras_total else 0.0

    first = group.iloc[0]
    return pd.Series({
        "RucProveedor": first["RucProveedor"],
        "CodProductoMaestro": first["CodProductoMaestro"],
        "ProductoMaestro": first.get("ProductoMaestro", ""),
        "CodProductoCpe": first["CodProductoCpe"],
        "ProductoCpe": first.get("ProductoCpe", ""),
        "UnidadMedidaCpe": first["UnidadMedidaCpe"],
        "UnidadMedidaCompra": first["UnidadMedidaCompra"],
        "FactorCantidadCompra": factor,
        "Muestras": muestras_total,
        "MuestrasModa": muestras_moda,
        "ConfianzaModa": round(confianza, 6),
    })


def construir_diccionario_conversion_unidades(conversion_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el diccionario final de conversión de cantidades.

    Cada fila histórica indica cuánta cantidad del CPE fue digitada como cantidad de compra.
    La regla aprendida es:

        CantidadCompraInferida = CantidadCpeFactura * FactorCantidadCompra

    Donde FactorCantidadCompra es la moda de CantidadCompra / CantidadCpe por:
    RUC + producto maestro + código CPE + unidad CPE + unidad compra.
    """
    if conversion_raw is None or conversion_raw.empty:
        return pd.DataFrame()

    df = _normalize_text_columns(conversion_raw)
    required = {
        "RucProveedor",
        "CodProductoMaestro",
        "CantidadCompra",
        "UnidadMedidaCompra",
        "CodProductoCpe",
        "CantidadCpe",
        "UnidadMedidaCpe",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "La consulta de conversión de unidades debe retornar las columnas: "
            f"{', '.join(sorted(required))}. Faltan: {', '.join(missing)}"
        )

    if "ProductoMaestro" not in df.columns:
        df["ProductoMaestro"] = ""
    if "ProductoCpe" not in df.columns:
        df["ProductoCpe"] = ""

    df["RucProveedor"] = df["RucProveedor"].map(_normalizar_ruc_conversion)
    df["CodProductoMaestro"] = df["CodProductoMaestro"].map(normalizar_codigo)
    df["CodProductoCpe"] = df["CodProductoCpe"].map(normalizar_codigo)
    df["UnidadMedidaCompra"] = df["UnidadMedidaCompra"].map(normalizar_unidad)
    df["UnidadMedidaCpe"] = df["UnidadMedidaCpe"].map(normalizar_unidad)
    df["CantidadCompra"] = pd.to_numeric(df["CantidadCompra"], errors="coerce").fillna(0.0)
    df["CantidadCpe"] = pd.to_numeric(df["CantidadCpe"], errors="coerce").fillna(0.0)

    df = df[
        (df["CodProductoMaestro"] != "")
        & (df["CodProductoCpe"] != "")
        & (df["CantidadCompra"] > 0.0)
        & (df["CantidadCpe"] > 0.0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["_factor_redondeado"] = (df["CantidadCompra"] / df["CantidadCpe"]).round(6)
    df = df[df["_factor_redondeado"] > 0.0].copy()

    if df.empty:
        return pd.DataFrame()

    group_cols = [
        "RucProveedor",
        "CodProductoMaestro",
        "CodProductoCpe",
        "UnidadMedidaCpe",
        "UnidadMedidaCompra",
    ]

    diccionario = (
        df.sort_values(group_cols + ["_factor_redondeado"])
        .groupby(group_cols, dropna=False)
        .apply(_modo_factor_conversion)
        .reset_index(drop=True)
    )

    return diccionario.sort_values(
        ["RucProveedor", "CodProductoMaestro", "CodProductoCpe", "UnidadMedidaCpe", "UnidadMedidaCompra"]
    ).reset_index(drop=True)


def generar_diccionario_conversion_unidades_raw(tenant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tenant_norm = normalizar_tenant(tenant)
    sql = _get_query_for_sede(tenant, "conversion_unidades")
    raw = _normalize_text_columns(_read_sql(sql, tenant))
    diccionario = construir_diccionario_conversion_unidades(raw)
    print(
        "Diccionario de conversión cargado para "
        f"tenant={tenant_norm}: raw_rows={len(raw)} dict_rows={len(diccionario)}"
    )
    return raw, diccionario


def generar_datasets_raw(tenant: str, overwrite: bool = True) -> dict:
    tenant_norm = normalizar_tenant(tenant)
    print(f"Generando datasets para tenant={tenant_norm}...")

    maestro_df = _normalize_text_columns(_read_sql(_get_query_for_sede(tenant, "maestro"), tenant))
    facturas_df = _normalize_text_columns(_read_sql(_get_query_for_sede(tenant, "facturas"), tenant))
    conversion_raw_df, conversion_dict_df = generar_diccionario_conversion_unidades_raw(tenant)

    raw_dir = get_tenant_raw_data_dir(tenant_norm)
    maestro_path = raw_dir / "maestro.csv"
    facturas_path = raw_dir / "productos_facturas.csv"
    conversion_raw_path = raw_dir / "conversion_unidades_raw.csv"
    conversion_dict_path = raw_dir / "diccionario_conversion_unidades.csv"

    if not overwrite:
        existentes = [
            str(path)
            for path in (maestro_path, facturas_path, conversion_raw_path, conversion_dict_path)
            if path.exists()
        ]
        if existentes:
            raise FileExistsError(
                "Ya existen datasets para este tenant y overwrite=False: "
                + ", ".join(existentes)
            )

    raw_dir.mkdir(parents=True, exist_ok=True)
    maestro_df.to_csv(maestro_path, index=False, encoding="utf-8-sig")
    facturas_df.to_csv(facturas_path, index=False, encoding="utf-8-sig")
    conversion_raw_df.to_csv(conversion_raw_path, index=False, encoding="utf-8-sig")
    conversion_dict_df.to_csv(conversion_dict_path, index=False, encoding="utf-8-sig")
    limpiar_cache_maestro(tenant_norm)

    return {
        "sede": tenant.upper(),
        "tenant": tenant_norm,
        "maestro": {
            "nombre": "maestro.csv",
            "filas": len(maestro_df),
            "ruta": str(maestro_path),
        },
        "productos_facturas": {
            "nombre": "productos_facturas.csv",
            "filas": len(facturas_df),
            "ruta": str(facturas_path),
        },
        "conversion_unidades_raw": {
            "nombre": "conversion_unidades_raw.csv",
            "filas": len(conversion_raw_df),
            "ruta": str(conversion_raw_path),
        },
        "diccionario_conversion_unidades": {
            "nombre": "diccionario_conversion_unidades.csv",
            "filas": len(conversion_dict_df),
            "ruta": str(conversion_dict_path),
        },
    }



def cargar_autohomologaciones(tenant: str) -> pd.DataFrame:
    """
    Lee las homologaciones inferidas desde las compras históricas del tenant.

    La consulta debe retornar, como mínimo:
      - CodProductoMaestro
      - CodProductoCpe

    Opcionalmente puede retornar:
      - RucProveedor
      - ProductoMaestro
      - ProductoCpe

    Esta salida se usa únicamente como verdad histórica para construir pares
    positivos durante el entrenamiento, no durante la inferencia.
    """
    tenant_norm = normalizar_tenant(tenant)
    sql = _get_query_for_sede(tenant, "auto_match")
    df = _normalize_text_columns(_read_sql(sql, tenant))

    required = {"CodProductoMaestro", "CodProductoCpe"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "La consulta de autohomologación debe retornar las columnas: "
            f"{', '.join(sorted(required))}. Faltan: {', '.join(missing)}"
        )

    print(f"Autohomologaciones cargadas para tenant={tenant_norm}: filas={len(df)}")
    return df
