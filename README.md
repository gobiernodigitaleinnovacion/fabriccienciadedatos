Portafolio de Ciencia de Datos en Microsoft Fabric

Bienvenido a mi repositorio de proyectos de ciencia de datos realizados en Microsoft Fabric. Este repositorio muestra ejercicios prácticos que demuestran mis habilidades en análisis de datos, machine learning y visualización.
Ejercicios

Crear, evaluar y puntuar un modelo de predicción de abandonoDesarrollé un modelo de machine learning para predecir el abandono de clientes de un banco usando un dataset con 1000 registros. El proceso incluyó:
Carga y limpieza: Carga de datos con Spark, eliminación de duplicados y columnas irrelevantes.
Visualización: Gráficos de barras e histogramas para explorar patrones de abandono.
Ingeniería de características: Creación de nuevas variables como NewTenure y discretización de CreditScore, Age y Balance.
Modelado: Entrené modelos Random Forest y LightGBM con SMOTE para manejar el desbalance de clases, usando MLflow para rastreo.
Evaluación: Comparé predicciones con matrices de confusión, logrando una precisión máxima con LightGBM (menos falsos positivos).
Análisis: Calculé tasas de abandono por geografía, género y otros factores.Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, LightGBM, MLflow, Seaborn, Matplotlib.Resultados: El modelo LightGBM tuvo el mejor rendimiento, con 105 falsos positivos y un equilibrio de precisión y recall.Ver notebook | Ver gráficos



Cómo navegar
Cada carpeta contiene:

Un notebook con el código completo, explicaciones y resultados.
Datasets utilizados (en formatos como CSV).
Gráficos generados (en PNG).
Tablas Delta (si aplica).

Los notebooks son ejecutables en Microsoft Fabric o en entornos con Python y las librerías indicadas (ver requisitos en cada notebook).
Contacto
[Tu Nombre] | LinkedIn | [Correo electrónico]
