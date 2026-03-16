# Motor de Homologación y Clasificación de Productos (DL_ProductMatchEngine)

Sistema de inteligencia aplicada para **homologar productos de facturas contra un maestro**, **clasificar atributos comerciales** y **reentrenar modelos de forma segura**, promoviendo un nuevo modelo solo cuando realmente supera al anterior.

## Descripción

Este proyecto automatiza el trabajo de normalización de productos entre fuentes heterogéneas, especialmente cuando:

- los códigos de producto no están homologados,
- el texto del producto viene sucio o inconsistente,
- existen múltiples proveedores con catálogos distintos,
- se necesita inferir atributos como marca, categoría, factor de venta y peso,
- se requiere mejorar el modelo con nuevos datos sin arriesgar el modelo productivo actual.

El sistema combina reglas basadas en maestro, procesamiento de texto y modelos de machine learning para resolver dos problemas principales:

1. **Clasificación de atributos del producto**
2. **Homologación entre producto factura y producto maestro**

---

## Capacidades principales

### 1) Clasificador de productos
Predice atributos del producto a partir de información de factura, como:

- `factorVenta`
- `factorConversion`
- `pesoUnitarioKg`
- `pesoCajaKg`
- `marca`
- `categoria`

Además, puede resolver varios de estos atributos directamente desde el **maestro** cuando existe evidencia suficiente, dejando la red neuronal como respaldo.

### 2) Homologador de productos
Encuentra el producto maestro más probable para un ítem de factura, incluso cuando:

- no existe coincidencia exacta por código,
- los nombres del producto tienen ruido,
- el proveedor usa una codificación distinta,
- hay diferencias de escritura, empaque o unidad.

### 3) Reentrenamiento seguro
El proyecto implementa una estrategia tipo **champion/challenger**:

- se entrena un modelo candidato,
- se evalúa contra la validación actual,
- se compara contra el modelo campeón vigente,
- **solo se promueve** si el candidato aprendió mejor,
- si no mejora, el modelo actual se conserva.

Esto evita degradar producción por un entrenamiento peor o inestable.

---

## Arquitectura funcional

El proyecto está dividido en dos motores principales:

### Clasificador
Responsable de inferir atributos del producto.

Componentes relevantes:

- construcción del dataset de entrenamiento,
- extracción y normalización de texto,
- generación de targets desde el maestro,
- entrenamiento del modelo,
- inferencia batch,
- resolución híbrida desde maestro para ciertos atributos.

### Homologador
Responsable de encontrar correspondencias entre producto factura y producto maestro.

Componentes relevantes:

- armado de pares positivos/negativos,
- entrenamiento inicial,
- minería de hard negatives,
- scoring de candidatos,
- ranking de matches.

### API
Se expone mediante **FastAPI**, con endpoints separados para:

- predicción del clasificador,
- predicción del homologador,
- entrenamiento del clasificador,
- entrenamiento del homologador.

### Gestión de artefactos
Los modelos y resultados de entrenamiento se organizan en carpetas de artefactos, incluyendo:

- campeón actual,
- candidatos,
- rechazados,
- modelos archivados reemplazados,
- reportes de entrenamiento y promoción.

---

## Flujo general

### Clasificación
1. Se recibe un lote de productos factura.
2. Se normaliza el texto y se construye el frame de inferencia.
3. El modelo predice atributos.
4. Si hay maestro disponible, algunos atributos pueden resolverse desde maestro.
5. Se devuelve una salida consolidada.

### Homologación
1. Se recibe un producto factura.
2. Se generan candidatos del maestro.
3. El modelo calcula score por candidato.
4. Se rankean los resultados.
5. Se devuelve el mejor match y/o top-k.

### Reentrenamiento
1. Se cargan datasets base.
2. Se construye el dataset de entrenamiento.
3. Se entrena un candidato.
4. Se evalúa candidato vs campeón sobre la misma validación actual.
5. Se decide si promover o rechazar.
6. Se archiva el resultado.

---

## Estructura esperada del proyecto

Estructura lógica referencial:

```text
app/
  api/
    v1/
      clasificador.py
      homologador.py
  schemas/
    clasificador.py
    homologador.py
  services/
    entrenamiento_clasificador_service.py
    entrenamiento_homologador_service.py
    inferencia_clasificador_service.py
    inferencia_homologacion_service.py

ml_pipeline/
  clasificador/
    model.py
    trainer.py
    training_data.py
    factor_resolver.py
    weight_resolver.py
  homologador/
    model.py
    trainer.py
  utils/
    retraining.py
    preparacion.py
    matching.py

scripts/
  clasificador/
    entrenar.py
  homologador/
    entrenar.py

data/
artifacts/