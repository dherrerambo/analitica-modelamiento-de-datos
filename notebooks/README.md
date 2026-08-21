# Contenido

  0. [Diagnóstico](./0_Diagnostic.ipynb)
  1. [Obtención de datos](./1_GetData.ipynb)
  2. [Análisis de datos exploratorio](./2_EDA.ipynb)
  3. [Ingeniería de características](./3_FeatureEngineering.ipynb)
  4. [Modelamiento de datos](./4_Modeling.ipynb)
  5. [Optimización de modelos](./5_ModelOptimization.ipynb)
  6. [Regresión](./6_Regression.ipynb)
  7. [Clustering](./7_Clustering.ipynb)
  8. [Redes Neuronales](./8_NN.ipynb)
  9. [Ética en el Modelamiento de Datos](./9_Ethics.ipynb)

Este README describe el proyecto desde la perspectiva **técnica**: qué lee y qué produce cada notebook, y con qué librerías. Para el contenido pedagógico del curso (objetivos de aprendizaje por sesión) ver el [README de la raíz del proyecto](../README.md).

## Resumen técnico por notebook

Antes de ejecutar los notebooks, configura el entorno de trabajo: [Preparación del Entorno de Trabajo en Python](#preparación-del-entorno-de-trabajo-en-python).

| # | Notebook | Lee de `../data`/`../models` | Escribe en `../models`/`../output` | Librerías/técnicas clave |
|---|---|---|---|---|
| 0 | `0_Diagnostic.ipynb` | — (autocontenido) | — (no persiste nada) | `assert` como autoverificación, sin dependencias externas |
| 1 | `1_GetData.ipynb` | — | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | `kagglehub`, crea `data/`, `models/`, `output/` |
| 2 | `2_EDA.ipynb` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | — (solo diagnóstico, no escribe archivos) | `pandas`, `seaborn`/`matplotlib` |
| 3 | `3_FeatureEngineering.ipynb` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | `data/preprocessed_data.csv` | Imputación, One-Hot Encoding, análisis de correlación |
| 4 | `4_Modeling.ipynb` | `data/preprocessed_data.csv` | `models/data.pkl`, `models/scaler.pkl`, `models/modelos_entrenados.pkl`, `models/metrics.pkl`, `output/Telco-Customer-Churn_Resultados.xlsx` | `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier` |
| 5 | `5_ModelOptimization.ipynb` | `models/data.pkl`, `models/metrics.pkl`, `models/modelos_entrenados.pkl`, `models/scaler.pkl` | `models/modelos_optimizados.pkl`, `models/metrics_modelos_optimizados.pkl`, actualiza el Excel de `4_Modeling.ipynb` | `GridSearchCV` |
| 6 | `6_Regression.ipynb` | `data/preprocessed_data.csv` | `models/regression_data.pkl`, `models/regression_modelos_entrenados.pkl`, `models/regression_metrics.pkl`, `models/{modelo}_regression.pkl` | `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor` |
| 7 | `7_Clustering.ipynb` | `data/preprocessed_data.csv` | `models/clustering_preprocesamiento.pkl`, `models/kmeans_clustering.pkl`, `models/dbscan_clustering.pkl`, `output/clusters_clientes.xlsx` | `KMeans`, `DBSCAN`, `PCA` (solo visualización), `NearestNeighbors` (calibración de `eps`) |
| 8 | `8_NN.ipynb` | `models/data.pkl`, `models/metrics.pkl`, `models/modelos_optimizados.pkl` | `models/nn_churn.keras`, `models/nn_metrics.pkl` | `tensorflow`/`keras` (`Sequential`, `Dense`, `Dropout`, `EarlyStopping`) |
| 9 | `9_Ethics.ipynb` | `models/data.pkl`, `models/modelos_entrenados.pkl` | — (notebook de análisis y discusión, no persiste artefactos) | Métricas desagregadas por subgrupo, sin librerías nuevas |

**Notas de lectura de la tabla:**
- `4_Modeling.ipynb` y `5_ModelOptimization.ipynb` guardan **dos versiones separadas** de los modelos clásicos: `modelos_entrenados.pkl` (hiperparámetros por defecto) y `modelos_optimizados.pkl` (afinados con `GridSearchCV`). `8_NN.ipynb` y `9_Ethics.ipynb` cargan una u otra según qué comparación necesiten — no son intercambiables.
- `modelos_entrenados.pkl` y `modelos_optimizados.pkl` tienen **estructuras distintas**: el primero guarda `{"model": estimador, "best_model": bool}` por modelo (hay que acceder con `info["model"]`); el segundo guarda el estimador directamente (`grid_search[name].best_estimator_`, se usa sin llave intermedia). Tenerlo presente al escribir un notebook nuevo que reutilice cualquiera de los dos.
- `6_Regression.ipynb` y `7_Clustering.ipynb` son independientes entre sí y de la rama de clasificación (4→5→8→9): ambos parten directamente de `preprocessed_data.csv`, no de los `.pkl` de `4_Modeling.ipynb`.

## Preparación del Entorno de Trabajo en Python
Este módulo tiene como objetivo preparar el entorno de desarrollo que se utilizará durante la especialización. Se recomienda seguir los pasos en el orden presentado.

### Requisitos
Instale las siguientes herramientas:
- **Python (versión estable recomendada, ver nota de compatibilidad más abajo)**. Ejemplo: [Python 3.12.x](https://www.python.org/downloads/)
- **Directorio recomendado de instalación de Python**. Crear una carpeta con el siguiente esquema:
  ```
  C:\python\pyXXXX\
  ```
  Donde `XXXX` corresponde a la versión instalada. Ejemplo:
  ```
  C:\python\py312\
  ```
- **IDE o editor de código**
    - [Visual Studio Code](https://code.visualstudio.com/) (recomendado)
    - [Antigravity](https://antigravity.google/download)
    - Jupyter Notebook (Anaconda)
    - Jupyter Labs
- **Control de versiones (opcional, pero recomendado)**: [Git para Windows](https://git-scm.com/install/windows)

> **Nota de compatibilidad — importante para `8_NN.ipynb`:** `tensorflow` (usado en el módulo de Redes Neuronales) actualmente soporta **Python 3.10 a 3.13**; todavía no soporta Python 3.14. Para poder ejecutar los 10 notebooks del curso con el mismo entorno virtual, se recomienda instalar **Python 3.12.x** en lugar de la última versión disponible. Verifica la [página de instalación de TensorFlow](https://www.tensorflow.org/install/pip) antes de crear el entorno, ya que este rango de versiones soportadas cambia con cada release.

### ¿Qué es un entorno virtual?
Un **entorno virtual** permite aislar las dependencias de cada proyecto, evitando conflictos entre versiones de librerías instaladas en el sistema.

#### Recomendaciones
- Utilizar siempre **versiones estables de Python**.
- Crear un entorno virtual por cada proyecto.
- No instalar librerías del proyecto directamente en la instalación global de Python.

### Crear y activar el entorno virtual

> Las instrucciones siguientes usan **CMD** (Símbolo del sistema). Si usa **PowerShell**, tenga en cuenta dos diferencias: el script de activación es `Activate.ps1` en lugar de `activate`, y la creación de alias se hace con `Set-Alias` en lugar de `doskey`.

Abra una terminal **CMD** y verifique la instalación de Python:
```cmd
python --version
```

Si el comando `python` no es reconocido, puede crear un alias temporal para la sesión actual de CMD:
```cmd
doskey python="C:\python\py312\python.exe" $*
```
> En PowerShell, el equivalente sería: `Set-Alias -Name python -Value "C:\python\py312\python.exe"`

Crear el entorno virtual:
```cmd
python -m venv analitica
```

También se puede definir una ruta específica donde se guardarán los archivos del entorno virtual:
```cmd
python -m venv "C:\Python\venvs\analitica"
```

Activar el entorno virtual:
```cmd
analitica\Scripts\activate
```
Si la activación fue exitosa, verá el prefijo `(analitica)` al inicio de la línea de comandos.

Si el entorno virtual se creó en una ruta específica, active usando la ruta completa, por ejemplo:
```cmd
C:\Python\venvs\analitica\Scripts\activate
```

Cuando existen varias versiones de Python instaladas, cree el entorno virtual indicando explícitamente el ejecutable deseado, para garantizar que se use la versión correcta:
```cmd
"C:\python\py312\python.exe" -m venv analitica
```

### Actualizar `pip`
Con el entorno virtual activado, ejecute:
```bash
python -m pip install --upgrade pip
```

### Instalar las dependencias del módulo
Instalación de las librerías principales:
```bash
pip install wheel jupyter ipykernel pandas numpy matplotlib seaborn scikit-learn==1.9.0 statsmodels openpyxl kagglehub tensorflow
```

> **Por qué `scikit-learn` va con versión fija:** los modelos ya entrenados que se
> publican en `models/*.pkl` se serializaron con la versión 1.9.0. `joblib.load()`
> advierte —y en algunos casos falla— al abrir un modelo guardado con una versión
> distinta a la instalada. Los notebooks del 4 al 9 repiten esta instalación en su
> primera celda, para que también funcionen en Google Colab, donde cada notebook
> arranca con un entorno nuevo.

### Registrar kernel para VSCode/Jupyter
```bash
"C:\Python\venvs\analitica\Scripts\python.exe" -m ipykernel install --user --name analitica --display-name "Python (analitica)"
```
Para su selección se debe tener en cuenta que aparecerá bajo `Select Another Kernel...` y posteriormente en `Jupyter Kernel...`.

#### Descripción rápida de las librerías
| Librería | Propósito |
|---|---|
| `wheel` | Soporte para paquetes precompilados |
| `jupyter` | Crear documentos interactivos que combinan código ejecutable, texto enriquecido y gráficos |
| `ipykernel` | Integración con Jupyter y VS Code |
| `pandas` | Manipulación y análisis de datos |
| `numpy` | Cálculo numérico vectorizado, base de `pandas` y `scikit-learn` |
| `matplotlib` | Visualización de datos (base gráfica usada por `seaborn`) |
| `seaborn` | Visualización estadística de datos |
| `scikit-learn` | Machine Learning (incluye `joblib` como dependencia, usado para guardar/cargar modelos) |
| `statsmodels` | Modelos estadísticos |
| `openpyxl` | Motor requerido por `pandas` para exportar resultados a archivos `.xlsx` |
| `kagglehub` | Descarga de datasets desde Kaggle |
| `tensorflow` | Construcción y entrenamiento de redes neuronales (módulo 8) |

### Verificar las librerías instaladas
```bash
pip list
```

### Desactivar el entorno virtual
Cuando termine de trabajar en el proyecto:
```cmd
deactivate
```

### Eliminar el entorno virtual
Si necesita recrearlo desde cero, elimine la carpeta `analitica`.

CMD:
```cmd
rmdir /s /q analitica
```

PowerShell:
```powershell
Remove-Item -Path .\analitica\ -Force -Recurse
```

### Estructura recomendada del proyecto
Este `README.md` vive dentro de `notebooks/`. Todos los notebooks del módulo referencian las demás carpetas con rutas relativas (`../data/`, `../models/`, `../output/`), por lo que deben ejecutarse siempre desde dentro de `notebooks/`.

> Las carpetas `data/`, `models/` y `output/` se crean automáticamente (si no existen) al ejecutar `1_GetData.ipynb`, el primer notebook del módulo. No es necesario crearlas manualmente.

```
analitica/
 │
 ├── notebooks/          # 0_Diagnostic ... 9_Ethics (este README)
 ├── data/               # dataset crudo y preprocesado (CSV)
 ├── models/             # modelos entrenados y artefactos serializados (.pkl)
 ├── output/             # resultados exportados (ej. Excel con predicciones)
 ├── README.md
 └── requirements.txt
```

### Generar el archivo de dependencias
Para facilitar la reproducción del entorno:
```bash
pip freeze > requirements.txt
```

Posteriormente, cualquier persona podrá instalar las mismas versiones con:
```bash
pip install -r requirements.txt
```

### Recomendación final
Antes de iniciar los módulos de analítica y modelamiento de datos, verifique que:
- `python --version` funciona correctamente.
- El entorno virtual está activado.
- `pip list` muestra las librerías instaladas.
- VS Code (u otro IDE) reconoce el intérprete ubicado en `analitica`.
- Al ejecutar `1_GetData.ipynb` por primera vez, se crean automáticamente las carpetas `data/`, `models/` y `output/` al mismo nivel que `notebooks/`.

Con esto el entorno quedará listo para trabajar con **Python, Jupyter, análisis de datos, estadística y machine learning** durante la especialización.
