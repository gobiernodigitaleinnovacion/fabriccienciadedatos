# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "ea6a2188-e8a0-47ad-a200-e41bffef534b",
# META       "default_lakehouse_name": "churn_lakehouse",
# META       "default_lakehouse_workspace_id": "dfe1c7ef-c511-43d8-a794-a1fbbbb49b7c",
# META       "known_lakehouses": [
# META         {
# META           "id": "ea6a2188-e8a0-47ad-a200-e41bffef534b"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# MAGIC %%writefile abfss://Fabric@onelake.dfs.fabric.microsoft.com/churn_lakehouse.Lakehouse/Files/requirements.txt
# MAGIC pandas>=1.5.0
# MAGIC pyspark>=3.4.0
# MAGIC scikit-learn>=1.3.2
# MAGIC imbalanced-learn>=0.13.0
# MAGIC lightgbm>=3.3.0
# MAGIC mlflow>=2.0.0
# MAGIC seaborn>=0.12.0
# MAGIC matplotlib>=3.7.0

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# MAGIC %%writefile abfss://Fabric@onelake.dfs.fabric.microsoft.com/churn_lakehouse.Lakehouse/Files/README.md
# MAGIC # Portafolio de Ciencia de Datos en Microsoft Fabric
# MAGIC 
# MAGIC Bienvenido a mi repositorio de proyectos de ciencia de datos realizados en Microsoft Fabric. Este repositorio muestra ejercicios prácticos que demuestran mis habilidades en análisis de datos, machine learning y visualización.
# MAGIC 
# MAGIC ## Ejercicios
# MAGIC 
# MAGIC - **[Crear, evaluar y puntuar un modelo de predicción de abandono](.)**  
# MAGIC   Desarrollé un modelo de machine learning para predecir el abandono de clientes de un banco usando un dataset con 1000 registros. El proceso incluyó:  
# MAGIC   - **Carga y limpieza**: Carga de datos con Spark, eliminación de duplicados y columnas irrelevantes (`RowNumber`, `CustomerId`, `Surname`).  
# MAGIC   - **Visualización**: Gráficos de barras e histogramas para explorar patrones de abandono por geografía, género, edad, etc.  
# MAGIC   - **Ingeniería de características**: Creación de variables como `NewTenure` (tenure/age) y discretización de `CreditScore`, `Age`, `Balance`, y `EstimatedSalary`.  
# MAGIC   - **Modelado**: Entrené dos modelos Random Forest (`max_depth=4` y `8`) y un modelo LightGBM con SMOTE para manejar el desbalance de clases, usando MLflow para rastreo.  
# MAGIC   - **Evaluación**: Comparé predicciones con matrices de confusión y métricas (precisión, recall, F1-score). LightGBM tuvo el mejor rendimiento, con 105 falsos positivos.  
# MAGIC   - **Análisis**: Calculé tasas de abandono por geografía (33.7% en Alemania vs. 17.2% en Francia), género, y otros factores, visualizadas en gráficos.  
# MAGIC   **Tecnologías**: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, LightGBM, MLflow, Seaborn, Matplotlib.  
# MAGIC   **Resultados**: El modelo LightGBM logró el mejor equilibrio de precisión y recall, con menos falsos positivos.  
# MAGIC   [Ver notebook](notebooks/churn_prediction.ipynb) | [Ver gráficos](results/)
# MAGIC 
# MAGIC ## Cómo navegar
# MAGIC 
# MAGIC Cada carpeta contiene:  
# MAGIC - Un notebook (`notebooks/`) con el código completo, explicaciones y resultados.  
# MAGIC - Datasets (`data/`) en formatos como CSV.  
# MAGIC - Gráficos (`results/`) en PNG, como matrices de confusión y tasas de abandono.  
# MAGIC - Un archivo `requirements.txt` con las dependencias.  
# MAGIC 
# MAGIC Los notebooks son ejecutables en Microsoft Fabric o en entornos con Python y las librerías indicadas. Consulta `requirements.txt` para instalar dependencias.
# MAGIC 
# MAGIC ## Requisitos
# MAGIC 
# MAGIC - Microsoft Fabric (para ejecución nativa) o Python 3.11 con Jupyter.  
# MAGIC - Dependencias: `pandas`, `pyspark`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `mlflow`, `seaborn`, `matplotlib` (ver `requirements.txt`).
# MAGIC 
# MAGIC ## Contacto
# MAGIC 
# MAGIC [Tu Nombre] | [LinkedIn](#) | [Correo electrónico]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
