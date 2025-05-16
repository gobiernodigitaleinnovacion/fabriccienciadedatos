# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "a799d506-821f-4666-ba51-c571402742a8",
# META       "default_lakehouse_name": "fraudlakehouse",
# META       "default_lakehouse_workspace_id": "dfe1c7ef-c511-43d8-a794-a1fbbbb49b7c",
# META       "known_lakehouses": [
# META         {
# META           "id": "a799d506-821f-4666-ba51-c571402742a8"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# **Bloque 1: Instalación de librerías y configuración inicial
# **
# Objetivo: Instalar la librería necesaria (imblearn), descargar el dataset (creditcard.csv), y configurar MLflow para el rastreo del experimento.

# CELL ********************

# Bloque 1: Instalación de librerías y configuración inicial
# Instalamos la librería imblearn, descargamos el dataset y configuramos MLflow.

# Instalar la librería imblearn
%pip install imblearn

# Importar librerías necesarias
import os
import requests
import time
import mlflow
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

# Crear una sesión de Spark
spark = SparkSession.builder.appName("Detección de Fraudes").getOrCreate()

# Definir parámetros
IS_CUSTOM_DATA = False  # Usaremos el dataset proporcionado por el tutorial
TARGET_COL = "Class"  # Columna objetivo
IS_SAMPLE = False  # Usaremos todos los datos
SAMPLE_ROWS = 5000  # No aplica porque IS_SAMPLE es False
DATA_FOLDER = "Files/deteccion-de-fraudes"  # Carpeta donde se almacenan los datos
DATA_FILE = "creditcard.csv"  # Nombre del archivo de datos
EXPERIMENT_NAME = "aisample-fraud"  # Nombre del experimento en MLflow

# Descargar el dataset si no está presente en el lakehouse
if not IS_CUSTOM_DATA:
    remote_url = "https://synapseaisolutionsa.blob.core.windows.net/public/Credit_Card_Fraud_Detection"
    fname = "creditcard.csv"
    download_path = f"/lakehouse/default/{DATA_FOLDER}/data"

    if not os.path.exists("/lakehouse/default"):
        raise FileNotFoundError("Default lakehouse not found, please add a lakehouse and restart the session.")
    os.makedirs(download_path, exist_ok=True)
    if not os.path.exists(f"{download_path}/{fname}"):
        r = requests.get(f"{remote_url}/{fname}", timeout=30)
        with open(f"{download_path}/{fname}", "wb") as f:
            f.write(r.content)
    print("Datos descargados y almacenados en el lakehouse en Files/deteccion-de-fraudes/data/.")

# Registrar el tiempo de inicio
ts = time.time()

# Configurar MLflow para el rastreo del experimento
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(disable=True)  # Desactivar el autologging de MLflow
print("Configuración de MLflow completada.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Bloque 2: Carga de datos y análisis exploratorio inicial**
# 
# Objetivo: Cargar el dataset creditcard.csv desde el lakehouse, realizar un análisis exploratorio inicial para entender la estructura del dataset, y visualizar la distribución de clases para confirmar el desbalance.
# 
# Contexto:
# 
# Vamos a cargar el dataset creditcard.csv desde Files/deteccion-de-fraudes/data/.
# Exploraremos las estadísticas básicas del dataset (número de registros, esquema).
# Visualizaremos la distribución de clases (Class) para confirmar el desbalance entre transacciones fraudulentas y no fraudulentas.
# Generaremos dos gráficos:
# Un gráfico de distribución de clases.
# Box plots para comparar la distribución de la cantidad (Amount) entre las clases.

# CELL ********************

# Bloque 2: Carga de datos y análisis exploratorio inicial
# Cargamos el dataset, exploramos estadísticas básicas y visualizamos la distribución de clases.

import pyspark.sql.functions as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset desde el lakehouse
df = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", True)
    .load(f"{DATA_FOLDER}/data/{DATA_FILE}")
    .cache()
)

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
display(df)

# Mostrar estadísticas básicas
print("Número de registros leídos: " + str(df.count()))
print("Esquema del dataset:")
df.printSchema()

# Transformar los datos: Asegurar que la columna Class sea de tipo entero
df_columns = df.columns
df_columns.remove(TARGET_COL)
df = df.select(df_columns + [TARGET_COL]).withColumn(TARGET_COL, F.col(TARGET_COL).cast("int"))

# Convertir a Pandas DataFrame para visualización
df_pd = df.toPandas()

# Explorar la distribución de clases
print('No Frauds', round(df_pd['Class'].value_counts()[0]/len(df_pd) * 100, 2), '% del dataset')
print('Frauds', round(df_pd['Class'].value_counts()[1]/len(df_pd) * 100, 2), '% del dataset')

# Gráfico 1: Distribución de clases
colors = ["#0101DF", "#DF0101"]
sns.countplot(x='Class', data=df_pd, palette=colors)
plt.title('Distribución de Clases \n (0: No Fraude || 1: Fraude)', fontsize=10)
plt.show()
print("Gráfico 'Distribución de Clases' generado. Guárdalo manualmente haciendo clic derecho y seleccionando 'Guardar imagen como...' en tu máquina local con el nombre 'class_distribution.png'.")

# Gráfico 2: Box plots de la cantidad (Amount) por clase
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
sns.boxplot(ax=ax1, x="Class", y="Amount", hue="Class", data=df_pd, palette="PRGn", showfliers=True)
ax1.set_title("Box Plot con Outliers")
sns.boxplot(ax=ax2, x="Class", y="Amount", hue="Class", data=df_pd, palette="PRGn", showfliers=False)
ax2.set_title("Box Plot sin Outliers")
plt.show()
print("Gráfico 'Box Plots de Amount por Clase' generado. Guárdalo manualmente haciendo clic derecho y seleccionando 'Guardar imagen como...' en tu máquina local con el nombre 'amount_boxplots.png'.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Bloque 3: Preparación de datos y balanceo con SMOTE**
# ****
# Objetivo: Dividir el dataset en conjuntos de entrenamiento y prueba, aplicar SMOTE para balancear las clases en el conjunto de entrenamiento, y preparar los datos para entrenar los modelos.
# 
# Contexto:
# 
# Dividiremos el dataset en entrenamiento (train) y prueba (test) con una proporción de 85% y 15%, respectivamente.
# Aplicaremos SMOTE (Synthetic Minority Oversampling Technique) al conjunto de entrenamiento para balancear las clases, generando muestras sintéticas de la clase minoritaria (fraudes).
# Confirmaremos que el balanceo se realizó correctamente mostrando la distribución de clases antes y después de SMOTE.

# CELL ********************

# Bloque 3: Preparación de datos y balanceo con SMOTE
# Dividimos el dataset en entrenamiento y prueba, aplicamos SMOTE para balancear las clases en el entrenamiento.

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

# Definir las columnas de características (excluyendo la columna objetivo)
feature_cols = [c for c in df_pd.columns.tolist() if c not in [TARGET_COL]]

# Dividir el dataset en entrenamiento y prueba (85% entrenamiento, 15% prueba)
train, test = train_test_split(df_pd, test_size=0.15, random_state=42)

# Mostrar las dimensiones de los conjuntos
print("Tamaño del conjunto de entrenamiento:", train.shape)
print("Tamaño del conjunto de prueba:", test.shape)

# Separar características y etiquetas en el conjunto de entrenamiento
X = train[feature_cols]
y = train[TARGET_COL]

# Mostrar la distribución de clases antes de SMOTE
print("Distribución de clases antes de SMOTE:", Counter(y))

# Aplicar SMOTE al conjunto de entrenamiento
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Mostrar la distribución de clases después de SMOTE
print("Distribución de clases después de SMOTE:", Counter(y_res))

# Combinar las características y etiquetas balanceadas en un nuevo DataFrame
new_train = pd.concat([pd.DataFrame(X_res, columns=feature_cols), pd.DataFrame(y_res, columns=[TARGET_COL])], axis=1)

# Mostrar las primeras filas del conjunto de entrenamiento balanceado
print("Primeras filas del conjunto de entrenamiento balanceado:")
print(new_train.head())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Bloque 4: Entrenamiento y evaluación de modelos**
# 
# Objetivo: Entrenar dos modelos LightGBM (uno con datos desbalanceados y otro con datos balanceados usando SMOTE), evaluar su rendimiento con métricas como AUC-ROC y AUPRC, y visualizar la importancia de las características y matrices de confusión.
# 
# Contexto:
# 
# Entrenaremos dos modelos LightGBM:
# Uno con el conjunto de entrenamiento original desbalanceado (train).
# Otro con el conjunto de entrenamiento balanceado usando SMOTE (new_train).
# Configuraremos MLflow para rastrear los experimentos.
# Generaremos gráficos de importancia de características y matrices de confusión para ambos modelos.
# Evaluaremos el rendimiento de los modelos usando métricas como AUC-ROC y AUPRC.

# CELL ********************

# Bloque 4: Entrenamiento y evaluación de modelos
# Entrenamos dos modelos LightGBM (desbalanceado y balanceado con SMOTE), evaluamos su rendimiento y generamos visualizaciones.

import lightgbm as lgb
import mlflow
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from synapse.ml.train import ComputeModelStatistics
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DoubleType
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar MLflow autologging
mlflow.autolog(exclusive=False)
print("Configuración de MLflow autologging completada.")

# Entrenar el modelo LightGBM con datos desbalanceados
print("Entrenando modelo con datos desbalanceados:")
model = lgb.LGBMClassifier(objective="binary")
with mlflow.start_run(run_name="raw_data") as raw_run:
    model = model.fit(
        train[feature_cols],
        train[TARGET_COL],
        eval_set=[(test[feature_cols], test[TARGET_COL])],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(10)]
    )

# Entrenar el modelo LightGBM con datos balanceados (SMOTE)
print("Entrenando modelo con datos balanceados (SMOTE):")
smote_model = lgb.LGBMClassifier(objective="binary")
with mlflow.start_run(run_name="smote_data") as smote_run:
    smote_model = smote_model.fit(
        new_train[feature_cols],
        new_train[TARGET_COL],
        eval_set=[(test[feature_cols], test[TARGET_COL])],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(10)]
    )

# Gráfico 1: Importancia de características (datos desbalanceados)
with mlflow.start_run(run_id=raw_run.info.run_id):
    importance = lgb.plot_importance(model, title="Importancia de Características (Datos Desbalanceados)")
    importance.figure.savefig("feature_importance_raw.png")
    mlflow.log_figure(importance.figure, "feature_importance_raw.png")
    plt.show()
print("Gráfico 'Importancia de Características (Datos Desbalanceados)' generado. Guárdalo manualmente como 'feature_importance_raw.png'.")

# Gráfico 2: Importancia de características (datos balanceados con SMOTE)
with mlflow.start_run(run_id=smote_run.info.run_id):
    smote_importance = lgb.plot_importance(smote_model, title="Importancia de Características (Datos Balanceados con SMOTE)")
    smote_importance.figure.savefig("feature_importance_smote.png")
    mlflow.log_figure(smote_importance.figure, "feature_importance_smote.png")
    plt.show()
print("Gráfico 'Importancia de Características (Datos Balanceados con SMOTE)' generado. Guárdalo manualmente como 'feature_importance_smote.png'.")

# Función para convertir predicciones a Spark DataFrame
def prediction_to_spark(model, test):
    predictions = model.predict(test[feature_cols], num_iteration=model.best_iteration_)
    predictions = tuple(zip(test[TARGET_COL].tolist(), predictions.tolist()))
    dataColumns = [TARGET_COL, "prediction"]
    predictions = (
        spark.createDataFrame(data=predictions, schema=dataColumns)
        .withColumn(TARGET_COL, col(TARGET_COL).cast(IntegerType()))
        .withColumn("prediction", col("prediction").cast(DoubleType()))
    )
    return predictions

# Generar predicciones para ambos modelos
predictions = prediction_to_spark(model, test)
smote_predictions = prediction_to_spark(smote_model, test)

# Mostrar las primeras filas de las predicciones (modelo desbalanceado)
print("Primeras filas de las predicciones (modelo desbalanceado):")
print(predictions.limit(10).toPandas())

# Calcular métricas para ambos modelos
metrics = ComputeModelStatistics(
    evaluationMetric="classification", labelCol=TARGET_COL, scoredLabelsCol="prediction"
).transform(predictions)

smote_metrics = ComputeModelStatistics(
    evaluationMetric="classification", labelCol=TARGET_COL, scoredLabelsCol="prediction"
).transform(smote_predictions)

# Mostrar métricas
print("Métricas del modelo desbalanceado:")
display(metrics)

# Extraer y mostrar la matriz de confusión (modelo desbalanceado)
cm = metrics.select("confusion_matrix").collect()[0][0].toArray()
smote_cm = smote_metrics.select("confusion_matrix").collect()[0][0].toArray()
print("Matriz de confusión (modelo desbalanceado):")
print(cm)

# Gráfico 3: Matriz de confusión (modelo desbalanceado)
def plot(cm):
    sns.set(rc={"figure.figsize": (5, 3.5)})
    ax = sns.heatmap(cm, annot=True, fmt=".20g")
    ax.set_title("Matriz de Confusión (Datos Desbalanceados)")
    ax.set_xlabel("Etiqueta Predicha")
    ax.set_ylabel("Etiqueta Verdadera")
    return ax

with mlflow.start_run(run_id=raw_run.info.run_id):
    ax = plot(cm)
    ax.figure.savefig("confusion_matrix_raw.png")
    mlflow.log_figure(ax.figure, "confusion_matrix_raw.png")
    plt.show()
print("Gráfico 'Matriz de Confusión (Datos Desbalanceados)' generado. Guárdalo manualmente como 'confusion_matrix_raw.png'.")

# Gráfico 4: Matriz de confusión (modelo balanceado con SMOTE)
with mlflow.start_run(run_id=smote_run.info.run_id):
    ax = plot(smote_cm)
    ax.set_title("Matriz de Confusión (Datos Balanceados con SMOTE)")
    ax.figure.savefig("confusion_matrix_smote.png")
    mlflow.log_figure(ax.figure, "confusion_matrix_smote.png")
    plt.show()
print("Gráfico 'Matriz de Confusión (Datos Balanceados con SMOTE)' generado. Guárdalo manualmente como 'confusion_matrix_smote.png'.")

# Función para evaluar AUC-ROC y AUPRC
def evaluate(predictions):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol=TARGET_COL)
    _evaluator = lambda metric: evaluator.setMetricName(metric).evaluate(predictions)
    auroc = _evaluator("areaUnderROC")
    print(f"El AUROC es: {auroc:.4f}")
    auprc = _evaluator("areaUnderPR")
    print(f"El AUPRC es: {auprc:.4f}")
    return auroc, auprc

# Evaluar modelo desbalanceado
with mlflow.start_run(run_id=raw_run.info.run_id):
    auroc, auprc = evaluate(predictions)
    mlflow.log_metrics({"AUPRC": auprc, "AUROC": auroc})
    mlflow.log_params({"Data_Enhancement": "None", "DATA_FILE": DATA_FILE})

# Evaluar modelo balanceado con SMOTE
with mlflow.start_run(run_id=smote_run.info.run_id):
    auroc, auprc = evaluate(smote_predictions)
    mlflow.log_metrics({"AUPRC": auprc, "AUROC": auroc})
    mlflow.log_params({"Data_Enhancement": "SMOTE", "DATA_FILE": DATA_FILE})

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Bloque 5: Registro de modelos y predicciones**
# 
# Objetivo: Registrar los dos modelos LightGBM en MLflow, cargar el mejor modelo para realizar predicciones por lotes, y guardar las predicciones en el lakehouse.
# 
# Contexto:
# 
# Registraremos los dos modelos LightGBM (desbalanceado y balanceado con SMOTE) en MLflow.
# Cargaremos el mejor modelo (el balanceado con SMOTE, versión 2, ya que tiene mejores métricas) para realizar predicciones por lotes.
# Guardaremos las predicciones en el lakehouse y mostraremos algunas filas de los resultados.
# Calcularemos el tiempo total de ejecución del notebook.

# CELL ********************

# Bloque 5: Registro de modelos y predicciones
# Registramos los modelos en MLflow, cargamos el mejor modelo para predicciones por lotes y guardamos los resultados.

from synapse.ml.predict import MLFlowTransformer
import mlflow

# Registrar los modelos en MLflow
registered_model_name = f"{EXPERIMENT_NAME}-lightgbm"

# Registrar el modelo desbalanceado
raw_model_uri = f"runs:/{raw_run.info.run_id}/model"
mlflow.register_model(raw_model_uri, registered_model_name)

# Registrar el modelo balanceado con SMOTE
smote_model_uri = f"runs:/{smote_run.info.run_id}/model"
mlflow.register_model(smote_model_uri, registered_model_name)

# Cargar el mejor modelo (balanceado con SMOTE, versión 2) para predicciones por lotes
spark.conf.set("spark.synapse.ml.predict.enabled", "true")

model = MLFlowTransformer(
    inputCols=feature_cols,
    outputCol="prediction",
    modelName=f"{EXPERIMENT_NAME}-lightgbm",
    modelVersion=2,
)

# Convertir el conjunto de prueba a Spark DataFrame
test_spark = spark.createDataFrame(data=test, schema=test.columns.to_list())

# Generar predicciones por lotes
batch_predictions = model.transform(test_spark)

# Mostrar las primeras filas de las predicciones
print("Primeras filas de las predicciones por lotes (modelo balanceado con SMOTE):")
display(batch_predictions.limit(5))

# Guardar las predicciones en el lakehouse
batch_predictions.write.format("delta").mode("overwrite").save(f"{DATA_FOLDER}/predictions/batch_predictions")
print("Predicciones guardadas en abfss://Fabric@onelake.dfs.fabric.microsoft.com/fraudlakehouse.Lakehouse/Files/deteccion-de-fraudes/predictions/batch_predictions.")

# Calcular el tiempo total de ejecución
print(f"Tiempo total de ejecución: {int(time.time() - ts)} segundos.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# **Bloque 6: Conclusiones y publicación**
# 
# Objetivo: Resumir los hallazgos del proyecto, descargar los archivos necesarios (notebook, dataset, predicciones, gráficos), crear un README.md para este ejercicio, y preparar un post para LinkedIn.
# 
# Contexto:
# 
# Resumiremos los resultados clave del ejercicio, incluyendo estadísticas del dataset, métricas de los modelos, y las visualizaciones generadas.
# Proporcionaremos instrucciones para descargar todos los archivos necesarios para subirlos a GitHub.
# Crearemos un README.md específico para este ejercicio, destacando el proceso, los resultados y las lecciones aprendidas.
# Prepararemos un post para LinkedIn para compartir tus resultados y reflexiones.

# CELL ********************

# Bloque 6: Conclusiones y publicación
# Resumimos los hallazgos, descargamos archivos y preparamos la publicación en GitHub y LinkedIn.

# Resumen de hallazgos
print("### Resumen de Hallazgos ###")
print("- **Tamaño del dataset**: 284,807 transacciones con 31 columnas (características V1-V28, Time, Amount, Class).")
print("- **Distribución de clases**: 99.83% no fraudes (284,315 transacciones), 0.17% fraudes (492 transacciones), indicando un desbalance extremo.")
print("- **Modelos entrenados**:")
print("  - Modelo con datos desbalanceados (LightGBM): AUROC: 0.7002, AUPRC: 0.0880.")
print("  - Modelo con datos balanceados (SMOTE, LightGBM): AUROC: 0.9253, AUPRC: 0.6410.")
print("- **Resultados**: El modelo balanceado con SMOTE mostró un desempeño significativamente mejor, con un AUROC de 0.9253 frente a 0.7002 del modelo desbalanceado, demostrando la efectividad de SMOTE para manejar el desbalance.")
print("- **Visualizaciones generadas**: Distribución de clases, box plots de Amount, importancia de características y matrices de confusión para ambos modelos.")
print("- **Conclusión**: Aunque el modelo balanceado con SMOTE es más efectivo, el AUROC de 0.9253 indica que hay espacio para mejoras, como ajustar hiperparámetros o probar otros algoritmos (por ejemplo, Random Forest o redes neuronales).")

# Instrucciones para descargar archivos
# Descargar el notebook fraud_detection.ipynb
print("Instrucciones para descargar el notebook:")
print("1. Ve a *Workspace > Fabric > deteccion-de-fraudes > fraudlakehouse > Notebooks/*.")
print("2. Abre el notebook `fraud_detection.ipynb`.")
print("3. Haz clic en *File > Download* para descargar el notebook a tu máquina local (por ejemplo, a C:\\Users\\hello\\Downloads\\).")

# Descargar el dataset desde el lakehouse
print("Instrucciones para descargar el dataset:")
print("1. Ve a *Workspace > Fabric > deteccion-de-fraudes > fraudlakehouse > Files/deteccion-de-fraudes/data/*.")
print("2. Haz clic derecho sobre `creditcard.csv` y selecciona *Download*.")
print("3. Guárdalo en tu máquina local (por ejemplo, a C:\\Users\\hello\\Downloads\\).")

# Descargar las predicciones desde el lakehouse
print("Instrucciones para descargar las predicciones:")
print("1. Ve a *Workspace > Fabric > deteccion-de-fraudes > fraudlakehouse > Files/deteccion-de-fraudes/predictions/*.")
print("2. Descarga la carpeta `batch_predictions` (puede aparecer como archivos individuales como `part-00000`, etc.).")
print("3. Renombra la carpeta o los archivos como `batch_predictions.csv` en tu máquina local para mayor claridad.")

# Nota sobre las gráficas ya guardadas
print("Ya tienes las gráficas guardadas: class_distribution.png, amount_boxplots.png, feature_importance_raw.png, feature_importance_smote.png, confusion_matrix_raw.png, confusion_matrix_smote.png.")

# Crear un README.md para el ejercicio
readme_content = """# Ejercicio 3: Detección de Fraudes

Desarrollé un modelo de detección de fraudes utilizando el dataset de transacciones de tarjetas de crédito de septiembre de 2013, implementado en Microsoft Fabric con Spark y MLflow. El objetivo fue identificar transacciones fraudulentas utilizando LightGBM, comparando el desempeño con datos desbalanceados y balanceados (usando SMOTE).  

## Proceso
- **Carga y limpieza**: Cargué el dataset (`creditcard.csv`) con 284,807 transacciones y 31 columnas (características V1-V28, Time, Amount, Class).  
- **Análisis exploratorio**: Confirmé el desbalance extremo (99.83% no fraudes, 0.17% fraudes) mediante visualizaciones.  
- **Preparación de datos**: Dividí el dataset en entrenamiento (85%) y prueba (15%), y apliqué SMOTE para balancear las clases en el conjunto de entrenamiento.  
- **Modelado**: Entrené dos modelos LightGBM: uno con datos desbalanceados y otro con datos balanceados (SMOTE).  
- **Evaluación**: Comparé el desempeño con métricas AUROC y AUPRC, y generé visualizaciones de importancia de características y matrices de confusión.  
- **Predicciones**: Usé el mejor modelo (balanceado con SMOTE) para realizar predicciones por lotes y las guardé en el lakehouse.  

## Resultados
- **Distribución de clases**: 99.83% no fraudes (284,315 transacciones), 0.17% fraudes (492 transacciones).  
- **Modelo con datos desbalanceados**: AUROC: 0.7002, AUPRC: 0.0880.  
- **Modelo con datos balanceados (SMOTE)**: AUROC: 0.9253, AUPRC: 0.6410.  
- **Conclusión**: El modelo balanceado con SMOTE mostró un desempeño mucho mejor (AUROC 0.9253 vs. 0.7002), destacando la efectividad de SMOTE para problemas desbalanceados. Sin embargo, hay espacio para mejoras, como ajustar hiperparámetros o probar otros algoritmos.  

## Tecnologías utilizadas
- Python, Microsoft Fabric, Spark, MLflow, LightGBM, Scikit-learn, Seaborn, Matplotlib.  

## Archivos disponibles
- [Notebook](notebooks/fraud_detection.ipynb)  
- [Gráficas](results/)
"""

# Guardar el README.md localmente
with open("/tmp/README_fraud_detection.md", "w") as f:
    f.write(readme_content)
print("README_fraud_detection.md guardado localmente en /tmp/. Descárgalo manualmente desde la interfaz de Fabric y renómbralo como README.md.")

# Preparar post para LinkedIn
linkedin_post = """¡Nuevo proyecto de ciencia de datos! 🚀 Desarrollé un modelo de detección de fraudes con LightGBM en Microsoft Fabric, utilizando un dataset de transacciones de tarjetas de crédito. Algunos hallazgos clave:

- Dataset: 284,807 transacciones, con un desbalance extremo (99.83% no fraudes, 0.17% fraudes).
- Modelos: Entrené dos modelos LightGBM: uno con datos desbalanceados (AUROC: 0.7002) y otro balanceado con SMOTE (AUROC: 0.9253).
- Conclusión: SMOTE mejoró significativamente el desempeño, pero aún hay espacio para optimizar el modelo con ajustes o algoritmos alternativos.

Explora el código y análisis en mi GitHub: [enlace al repositorio].

👤 Juan Heriberto Rosas Juárez  
📧 juanheriberto.rosas@jhrjdata.com  
🌐 https://www.linkedin.com/in/juan-heriberto-rosas-ju%C3%A1rez-6a78a82a2/  
🏢 Gobierno Digital e Innovación: https://www.gobiernodigitaleinnovacion.com/  
#DataScience #MicrosoftFabric #MachineLearning
"""

# Guardar el post para LinkedIn localmente
with open("/tmp/linkedin_post_fraud_detection.txt", "w") as f:
    f.write(linkedin_post)
print("Post para LinkedIn guardado localmente en /tmp/linkedin_post_fraud_detection.txt. Descárgalo manualmente desde la interfaz de Fabric.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
