# Portafolio de Ciencia de Datos en Microsoft Fabric

Bienvenido a mi repositorio de proyectos de ciencia de datos realizados en Microsoft Fabric. Este repositorio contiene una colección de ejercicios prácticos end-to-end que demuestran mis habilidades en análisis de datos, machine learning y visualización. Cada proyecto abarca desde la carga y limpieza de datos hasta el modelado, evaluación y análisis de resultados, utilizando herramientas modernas de ciencia de datos.

## Ejercicios

### Crear, evaluar y puntuar un modelo de predicción de abandono
Desarrollé un modelo de machine learning para predecir el abandono de clientes de un banco usando un dataset con 10,000 registros. El proceso incluyó:  
- **Carga y limpieza**: Carga de datos con Spark, eliminación de duplicados y columnas irrelevantes (`RowNumber`, `CustomerId`, `Surname`).  
- **Visualización**: Gráficos de barras e histogramas para explorar patrones de abandono por geografía, género, edad, etc.  
- **Ingeniería de características**: Creación de variables como `NewTenure` (tenure/age) y discretización de `CreditScore`, `Age`, `Balance`, y `EstimatedSalary`.  
- **Modelado**: Entrené dos modelos Random Forest (`max_depth=4` y `8`) y un modelo LightGBM con SMOTE para manejar el desbalance de clases, usando MLflow para rastreo.  
- **Evaluación**: Comparé predicciones con matrices de confusión y métricas (precisión, recall, F1-score). LightGBM tuvo el mejor rendimiento, con 74% de precisión para la clase de abandono.  
- **Análisis**: Calculé tasas de abandono por geografía (32.51% en Alemania vs. 16.34% en Francia), género (25.07% mujeres vs. 16.46% hombres), y otros factores.  
**Tecnologías**: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, LightGBM, MLflow, Seaborn, Matplotlib.  
**Resultados**: LightGBM logró el mejor equilibrio de precisión y recall, con menos falsos positivos.  
[Ver notebook](Portafolio-de-Ciencia-de-Datos-en-Microsoft-Fabric/Ejercicio-1/notebooks/churn_prediction.ipynb)

*Más ejercicios serán añadidos próximamente.*

## Cómo navegar
Cada carpeta de ejercicio contiene:  
- Un notebook (`notebooks/`) con el código completo, explicaciones y resultados.  
- Datasets utilizados (`data/`) en formatos como CSV.  
- Gráficos generados (`results/`) en PNG, como matrices de confusión y tasas de abandono (si están disponibles).  
- Tablas Delta (si aplica).  

Los notebooks son ejecutables en Microsoft Fabric o en entornos con Python y las librerías indicadas (ver requisitos en cada notebook).

## Requisitos
- Microsoft Fabric (para ejecución nativa) o Python 3.11 con Jupyter.  
- Dependencias comunes: `pandas`, `pyspark`, `scikit-learn`, `imbalanced-learn`, `lightgbm`, `mlflow`, `seaborn`, `matplotlib` (ver `requirements.txt` en cada ejercicio).

## Contacto
Juan Heriberto Rosas Juárez | [LinkedIn](https://www.linkedin.com/in/juan-heriberto-rosas-ju%C3%A1rez-6a78a82a2/) | [Correo electrónico](mailto:juanheriberto.rosas@jhrjdata.com)  
Empresa: [Gobierno Digital e Innovación](https://www.gobiernodigitaleinnovacion.com/)
