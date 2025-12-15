# Trabajo Práctico N°3 – Procesamiento de Imágenes I

**Tecnicatura Universitaria en Inteligencia Artificial (TUIA)**  
**Facultad de Ciencias Exactas, Ingeniería y Agrimensura – UNR**  
**Procesamiento de Imágenes I (IA 4.4)**

**Autores:**
- Sebastián Pérez
- Tomás Calabozo
- Tomás Lapolla

**Año:** 2025 – 2° Cuatrimestre

---

## Descripción general

Este repositorio contiene el desarrollo completo del **Trabajo Práctico N°3** de la materia **Procesamiento de Imágenes I**, correspondiente a la Tecnicatura Universitaria en Inteligencia Artificial (TUIA – UNR).

El objetivo del trabajo es diseñar e implementar un **pipeline de procesamiento de imágenes y video** capaz de **detectar dados rojos en una escena, determinar el valor de cada dado (cantidad de puntos) y calcular la suma total**, utilizando técnicas vistas a lo largo de la cursada.

Se pone especial énfasis en:
- la **visualización de cada etapa del procesamiento**,
- la validación visual de resultados,
- el análisis del comportamiento del sistema sobre secuencias de video.

El desarrollo se acompaña de un **informe final en PDF**, donde se documenta en detalle el funcionamiento del algoritmo, los parámetros elegidos y los resultados obtenidos.

---

## Enfoque y metodología

El algoritmo se estructura en una secuencia de etapas claras:

1. **Preprocesamiento**
   - Conversión de espacio de color (BGR → HSV).
   - Acondicionamiento general para facilitar segmentación y conteo.

2. **Segmentación de dados (por color)**
   - Se filtran los **tonos rojos** en HSV utilizando dos rangos (debido a la naturaleza circular del canal H).
   - Se combinan ambas máscaras y se aplican operaciones morfológicas:
     - **Clausura** para rellenar regiones de dado.
     - **Apertura** para eliminar ruido.

3. **Extracción de contornos y filtrado**
   - Se detectan contornos externos sobre la máscara.
   - Se filtran por área para conservar candidatos consistentes con el tamaño esperado de un dado.
   - Para cada contorno válido se calcula:
     - `boundingRect` → `(x, y, w, h)`
     - centro aproximado `(cx, cy)`
     - región de interés `ROI` recortada del frame
   - Se agrega un margen fijo (**padding**) al bounding box para mejorar visualización y evitar recortes al borde.

4. **Estructura `info_dados`**
   - Como salida de la segmentación se construye una lista `info_dados` con, por cada dado detectado:
     - `imagen` (ROI),
     - `bounding_box`,
     - `centro`,
     - `area`,
     - `puntos` (inicialmente 0).

5. **Detección de reposo (estabilización temporal)**
   - Se compara el movimiento entre centroides actuales y previos.
   - Si el movimiento total se mantiene debajo de un umbral durante varios frames consecutivos, se considera que los dados están **en reposo**.
   - El conteo de puntos se realiza **solo cuando hay reposo**, para evitar errores por motion blur o cambios de forma.

6. **Detección del valor del dado (conteo de puntos)**
   - Para cada ROI:
     - Se segmentan los puntos blancos en HSV.
     - Se detectan contornos de los puntos y se filtran por:
       - área mínima (ruido),
       - rango de área relativo al tamaño de la ROI,
       - validación geométrica mediante un **factor de forma** (aproximación a circularidad).
   - Se devuelve la cantidad de puntos detectados (entre 1 y 6). Si se obtiene un número inválido, se retorna 0.

7. **Visualización y salida**
   - Se dibujan bounding boxes, IDs y (cuando corresponde) el valor detectado sobre cada dado.
   - Se muestra una barra superior con el estado del análisis (contador de reposo, cantidad de dados, etc.).
   - Se imprime por consola el resultado final de cada reposo: valores individuales y suma total.

---

## Contenido del repositorio

- **Informe final (PDF)**  
  Documento principal donde se detalla el desarrollo completo del trabajo, fundamentos, explicación de cada etapa, resultados y conclusiones.

- **Código fuente (Python)**  
  Implementación del pipeline utilizando principalmente:
  - OpenCV
  - NumPy
  - Matplotlib

- **Salidas opcionales (frames / videos procesados)**  
  Generadas automáticamente cuando se habilita el guardado mediante una bandera del `main`.

---

## Configuración del script y uso de banderas (main)

El comportamiento del script principal se controla mediante banderas definidas en el bloque:

```python
if __name__ == '__main__':
    MAIN = True
    GUARDAR_DATOS = False
    MOSTRAR_DETALLE_PUNTOS = True
    HIST = True
