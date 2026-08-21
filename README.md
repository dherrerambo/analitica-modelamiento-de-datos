# Análisis y Modelamiento de Datos

Especialización en Analítica de Datos — seminario de Analítica y Modelamiento de Datos.

Este README describe el curso desde la perspectiva de **contenido y aprendizaje**: qué se cubre en cada sesión, qué se espera que el estudiante entienda y sea capaz de hacer al terminarla, y qué caso de negocio la conecta con las demás. Para el detalle técnico de cada notebook (entradas, salidas, librerías) ver el [README de `notebooks/`](./notebooks/README.md).

## Contenido
1. [Bienvenida](#1-bienvenida)
2. [Requisitos](#2-requisitos)
3. [Diagnóstico — Notebook 0](#3-diagnóstico)
4. [Dataset: Telco Customer Churn — Notebook 1](#4-dataset-telco-customer-churn)
5. [Análisis de datos exploratorio (EDA) — Notebook 2](#5-análisis-de-datos-exploratorio-eda)
6. [Feature Engineering (Ingeniería de características) — Notebook 3](#6-feature-engineering-ingeniería-de-características)
7. [Modelamiento de datos — Notebook 4](#7-modelamiento-de-datos)
8. [Optimización de modelos — Notebook 5](#8-optimización-de-modelos)
9. [Regresión — Notebook 6](#9-regresión)
10. [Clustering — Notebook 7](#10-clustering)
11. [Redes Neuronales — Notebook 8](#11-redes-neuronales)
12. [Ética en el Modelamiento de Datos — Notebook 9](#12-ética-en-el-modelamiento-de-datos)

## 1. Bienvenida

### ¿Qué NO es este seminario?
* Un curso de programación desde cero.
* Un curso de matemáticas avanzadas.
* Un curso de inteligencia artificial profunda.

### ¿Qué SÍ es?
* Analítica aplicada a problemas de negocio.
* Modelamiento predictivo práctico.
* Interpretación y comunicación de resultados.
* Toma de decisiones basada en datos.

### Hilo conductor del curso
Los 10 notebooks (0 a 9) no son ejercicios sueltos: comparten **un solo caso de negocio** (predicción de fuga de clientes — *churn* — en una empresa de telecomunicaciones) y **un solo dataset**, y cada notebook consume artefactos que dejó el anterior. La progresión sigue el ciclo de vida real de un proyecto analítico:

```
Diagnóstico → Datos → Exploración → Preparación → Modelado → Optimización
   → (Regresión / Clustering / Redes Neuronales, en paralelo, mismo dataset)
   → Ética y gobernanza
```

## 2. Requisitos
* Python 3.10+ (ver nota de compatibilidad con `tensorflow` en el [README de `notebooks/`](./notebooks/README.md))
* IDE instalado (Jupyter Notebook, Jupyter Lab, VS Code, Antigravity, etc.)
* [Preparación del Entorno de Trabajo en Python](./notebooks/README.md)
* Power BI (opcional)

### Uso en Colab
Ejecuta estas dos celdas al inicio de **cada** sesión, antes de correr el notebook:

~~~bash
!git clone https://github.com/dherrerambo/analitica-modelamiento-de-datos
%cd analitica-modelamiento-de-datos/notebooks
~~~

~~~bash
!pip install wheel pandas numpy matplotlib seaborn scikit-learn==1.9.0 statsmodels openpyxl kagglehub tensorflow
~~~

> **El `%cd` no es opcional.** Los notebooks leen y escriben con rutas relativas (`../data/`, `../models/`), así que deben ejecutarse desde dentro de `notebooks/`. Sin ese cambio de directorio fallan a partir de `1_GetData.ipynb`.

> **Colab borra todo al cerrar la sesión.** El clon y los archivos que generes (`models/*.pkl`, `output/*.xlsx`) viven en una máquina temporal que se descarta al cerrar la pestaña o tras un rato de inactividad. No es un problema: en la siguiente sesión vuelve a ejecutar las dos celdas y el `git clone` traerá la versión más reciente del repositorio, con los módulos publicados hasta ese momento.
>
> Tus respuestas sí se conservan, siempre que uses **Archivo → Guardar una copia en Drive**. Esa copia vive en tu Google Drive y es independiente del repositorio: trabaja siempre sobre ella, nunca sobre el archivo que quedó dentro del clon.


## 3. Diagnóstico
Antes de iniciar el módulo de negocio, cada estudiante hace un diagnóstico de su nivel en cuatro frentes: fundamentos de Python, estadística descriptiva, manipulación de datos con `pandas` y conceptos generales de Machine Learning. No es un examen calificable: es un mapa de brechas para calibrar el ritmo de las primeras sesiones, y un punto de comparación contra el cierre del curso.

**Al terminar esta sesión el estudiante puede:** ubicar sus propias brechas de conocimiento antes de que estas se conviertan en un obstáculo en los notebooks técnicos.

Notebook: [***0_Diagnostic.ipynb***](./notebooks/0_Diagnostic.ipynb)

## 4. Dataset: Telco Customer Churn
Se presenta el caso de negocio que atraviesa todo el curso — una empresa de telecomunicaciones que quiere anticipar qué clientes están en riesgo de cancelar el servicio — y se descarga el dataset público que lo representa.

* Fuente oficial (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
* Nombre del archivo principal: ***WA_Fn-UseC_-Telco-Customer-Churn.csv***

**Al terminar esta sesión el estudiante puede:** obtener un dataset de forma reproducible (sin descargas manuales) y dejarlo disponible para el resto del proyecto, con la estructura de carpetas del proyecto ya creada.

Notebook: [***1_GetData.ipynb***](./notebooks/1_GetData.ipynb)

## 5. Análisis de datos exploratorio (EDA)
El **Análisis Exploratorio de Datos (EDA)** es la fase del proceso analítico en la que buscamos comprender los datos antes de limpiarlos, transformarlos o construir modelos predictivos. El objetivo principal no es obtener conclusiones definitivas, sino formular hipótesis, detectar problemas de calidad y entender el comportamiento general de la información.

**Al terminar esta sesión el estudiante puede:** diagnosticar problemas de calidad de datos (valores no convertibles, nulos), y sustentar con evidencia visual qué variables parecen asociadas al churn, antes de tocar un solo modelo.

Notebook: [***2_EDA.ipynb***](./notebooks/2_EDA.ipynb)

## 6. Feature Engineering (Ingeniería de características)
La ingeniería de características es la etapa del proceso analítico en la que transformamos, combinamos o construimos variables con el objetivo de representar mejor el fenómeno de negocio y preparar los datos para el modelamiento analítico.

**Al terminar esta sesión el estudiante puede:** justificar y ejecutar decisiones de limpieza (imputación con criterio de negocio, no automática), y transformar variables categóricas y numéricas a un formato listo para cualquier algoritmo de `scikit-learn`.

Notebook: [***3_FeatureEngineering.ipynb***](./notebooks/3_FeatureEngineering.ipynb)

## 7. Modelamiento de datos
Es la etapa en la que se entrenan y comparan distintos modelos de clasificación para predecir el churn, se evalúa su desempeño con métricas apropiadas al problema de negocio, y se selecciona y persiste el modelo con mejor desempeño para su reutilización.

**Al terminar esta sesión el estudiante puede:** entrenar y comparar varios algoritmos de clasificación bajo un mismo protocolo (misma partición, mismas métricas), elegir el mejor con criterio justificado (no solo por accuracy), y dejar un modelo listo para producción.

Notebook: [***4_Modeling.ipynb***](./notebooks/4_Modeling.ipynb)

## 8. Optimización de modelos
Los modelos del módulo anterior se entrenaron con hiperparámetros por defecto. Aquí se busca sistemáticamente la mejor combinación de hiperparámetros mediante **Grid Search**, y se compara el desempeño antes y después de la optimización.

**Al terminar esta sesión el estudiante puede:** distinguir un parámetro de un hiperparámetro, y ajustar sistemáticamente un modelo en lugar de aceptar los valores por defecto de la librería.

Notebook: [***5_ModelOptimization.ipynb***](./notebooks/5_ModelOptimization.ipynb)

## 9. Regresión
Cambia el tipo de problema de clasificación a **regresión**, reutilizando el mismo dataset ya conocido: se estima un valor numérico continuo (el cargo mensual del cliente, `MonthlyCharges`) en lugar de una categoría, con sus propios algoritmos (Regresión Lineal, árboles de regresión) y métricas (R², MAE, RMSE).

**Al terminar esta sesión el estudiante puede:** reconocer cuándo un problema es de clasificación y cuándo es de regresión, y adaptar el mismo flujo de trabajo (partición, entrenamiento, evaluación) a un tipo de problema distinto.

Notebook: [***6_Regression.ipynb***](./notebooks/6_Regression.ipynb)

## 10. Clustering
Introduce el aprendizaje **no supervisado**: agrupar clientes por similitud sin una variable objetivo, usando **K-Means** (partición) y **DBSCAN** (densidad) sobre el mismo dataset Telco Customer Churn.

**Al terminar esta sesión el estudiante puede:** segmentar clientes sin depender de una etiqueta previa, y traducir un cluster estadístico en un segmento de negocio interpretable.

Notebook: [***7_Clustering.ipynb***](./notebooks/7_Clustering.ipynb)

## 11. Redes Neuronales
Introduce Deep Learning con **TensorFlow**/**Keras**, construyendo un Perceptrón Multicapa (MLP) sobre el mismo caso de negocio (predicción de churn) ya trabajado en los módulos de clasificación, comparándolo directamente contra los modelos clásicos ya optimizados en el módulo anterior.

**Al terminar esta sesión el estudiante puede:** explicar qué es una red neuronal a nivel conceptual, entrenarla evitando sobreajuste, y argumentar quando un modelo clásico sigue siendo preferible a una red neuronal (no siempre "más complejo" es "mejor").

> Requiere Python 3.10–3.13 (ver nota de compatibilidad en el [README de `notebooks/`](./notebooks/README.md)).

Notebook: [***8_NN.ipynb***](./notebooks/8_NN.ipynb)

## 12. Ética en el Modelamiento de Datos
No introduce un algoritmo nuevo: reutiliza los modelos ya entrenados para examinarlos desde otra perspectiva — sesgo y equidad por subgrupo demográfico, minimización de datos personales (GDPR), transparencia/explicabilidad, y gobernanza del ciclo de vida del modelo (ISO/IEC 27001). Cierra con una checklist ética reutilizable para futuros proyectos.

**Al terminar esta sesión el estudiante puede:** auditar un modelo ya construido en busca de sesgos no evidentes, y justificar decisiones de diseño (qué variables usar, qué tan explicable debe ser un modelo) más allá del desempeño técnico.

Notebook: [***9_Ethics.ipynb***](./notebooks/9_Ethics.ipynb)
