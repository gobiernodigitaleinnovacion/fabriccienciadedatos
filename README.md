Portafolio de Ciencia de Datos en Microsoft Fabric 📊
Bienvenido a mi repositorio de proyectos de ciencia de datos realizados en Microsoft Fabric. Este portafolio contiene una colección de 8 ejercicios prácticos end-to-end que demuestran mis habilidades en análisis de datos, machine learning, series temporales y visualización. Cada proyecto abarca desde la carga y limpieza de datos hasta el modelado, evaluación y análisis de resultados, utilizando herramientas modernas de ciencia de datos en un entorno distribuido.
🎯 Objetivo
El objetivo de este portafolio es mostrar mi capacidad para resolver problemas de negocio reales mediante técnicas de ciencia de datos, aplicando un flujo completo que incluye exploración de datos, preprocesamiento, modelado, evaluación y visualización. Los proyectos cubren diversas áreas, como predicción de abandono, recomendación de productos, detección de fraudes, pronósticos de series temporales, clasificación de texto, modelado de uplift, detección de fallos y pronóstico de ventas.
📂 Estructura del Repositorio
Cada ejercicio sigue una estructura estándar para facilitar la navegación:

notebooks/: Notebook con el código completo, explicaciones y resultados.
data/: Datasets utilizados (CSV, Excel, etc.).
results/: Gráficas generadas (PNG), como visualizaciones de datos y resultados de modelos.
Tablas Delta: Resultados almacenados en el lakehouse (si aplica).
README.md: Descripción detallada de cada ejercicio.

📈 Ejercicios
1. Predicción de Abandono de Clientes
Desarrollé un modelo de machine learning para predecir el abandono de clientes de un banco usando un dataset con 10,000 registros.

Carga y limpieza: Cargué datos con Spark, eliminé duplicados y columnas irrelevantes (RowNumber, CustomerId, Surname).
Visualización: Gráficos de barras e histogramas para explorar patrones de abandono por geografía, género, edad, etc.
Ingeniería de características: Creé variables como NewTenure (tenure/age) y discretizé CreditScore, Age, Balance, y EstimatedSalary.
Modelado: Entrené dos modelos Random Forest (max_depth=4 y 8) y un modelo LightGBM con SMOTE para manejar el desbalance, usando MLflow para rastreo.
Evaluación: Comparé predicciones con matrices de confusión y métricas (precisión, recall, F1-score). LightGBM tuvo el mejor rendimiento, con 74% de precisión para la clase de abandono.
Análisis: Calculé tasas de abandono por geografía (32.51% en Alemania vs. 16.34% en Francia), género (25.07% mujeres vs. 16.46% hombres), y otros factores, visualizadas en gráficos.
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, LightGBM, MLflow, Seaborn, Matplotlib.
Resultados: LightGBM logró el mejor equilibrio de precisión y recall, con menos falsos positivos.Ver notebook | Ver gráficos


2. Sistema de Recomendación de Productos (Versión Inicial)
Construí un sistema de recomendación basado en popularidad y filtrado colaborativo para recomendar productos a clientes usando un dataset de 10,000 transacciones.

Carga y limpieza: Cargué datos con Spark, eliminé transacciones incompletas.
Análisis exploratorio: Visualicé los productos más vendidos y patrones de compra por cliente.
Modelado: Implementé un modelo de recomendación basado en popularidad (top 5 productos) y un modelo de filtrado colaborativo usando SVD (Singular Value Decomposition).
Evaluación: Comparé las recomendaciones con métricas como precisión@k (0.65 para SVD) y cobertura de catálogo.
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, Surprise (SVD), Matplotlib.
Resultados: El modelo SVD ofreció recomendaciones personalizadas, pero la cobertura fue limitada debido a la sparsity del dataset.Ver notebook | Ver gráficos


3. Sistema de Recomendación de Productos (Versión Mejorada)
Mejoré el sistema de recomendación inicial incorporando técnicas avanzadas y un enfoque híbrido.

Carga y preprocesamiento: Añadí más datos (15,000 transacciones) y normalicé las calificaciones implícitas.
Análisis exploratorio: Visualicé patrones de compra por segmento de cliente y categoría de producto.
Modelado: Combiné filtrado colaborativo (SVD) con un modelo basado en contenido (KNN con características de producto), creando un sistema híbrido.
Evaluación: Precisión@k mejoró a 0.78, con mayor cobertura y personalización.
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, Surprise, Matplotlib, Seaborn.
Resultados: El enfoque híbrido mejoró significativamente las recomendaciones, especialmente para usuarios nuevos (cold start).Ver notebook | Ver gráficos


4. Detección de Fraudes en Transacciones
Desarrollé un modelo de clasificación para detectar transacciones fraudulentas usando un dataset de 284,807 transacciones financieras.

Carga y limpieza: Cargué datos con Spark, manejé valores nulos y escalé características numéricas.
Análisis exploratorio: Visualicé el desbalance extremo (0.17% fraudes) y correlaciones entre características PCA.
Modelado: Entrené un modelo Isolation Forest y un modelo de Regresión Logística con SMOTE para balancear clases, usando MLflow para rastreo.
Evaluación: Isolation Forest logró un AUC-ROC de 0.92, mientras que la Regresión Logística alcanzó 0.89 con mejor recall para fraudes.
Análisis: Identifiqué patrones de fraude relacionados con montos altos y tiempos específicos, visualizados en gráficos.
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, Imbalanced-learn, MLflow, Seaborn, Matplotlib.
Resultados: Isolation Forest fue más efectivo para detectar anomalías, pero la Regresión Logística ofreció mejor interpretabilidad.Ver notebook | Ver gráficos


5. Pronóstico de Series Temporales
Construí un modelo de pronóstico para predecir ventas mensuales usando un dataset de series temporales con 5 años de datos.

Carga y preprocesamiento: Cargué datos con Spark, resampling a nivel mensual y manejé valores nulos.
Análisis exploratorio: Visualicé tendencias y estacionalidad, descomponiendo la serie en componentes (tendencia, estacionalidad, residuales).
Modelado: Entrené un modelo ARIMA ajustando parámetros con AIC, y un modelo Prophet para comparación.
Evaluación: ARIMA logró un MAPE de 12%, mientras que Prophet alcanzó 10.5%.
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Statsmodels (ARIMA), Prophet, Matplotlib.
Resultados: Prophet fue más preciso y manejó mejor la estacionalidad, ideal para pronósticos a largo plazo.Ver notebook | Ver gráficos


6. Clasificación de Texto
Desarrollé un modelo de clasificación de texto para identificar reseñas positivas y negativas usando un dataset de 50,000 reseñas.

Carga y preprocesamiento: Cargué datos con Spark, apliqué limpieza de texto (tokenización, eliminación de stop words, lematización).
Análisis exploratorio: Visualicé la distribución de clases y las palabras más frecuentes con nubes de palabras.
Modelado: Entrené un modelo de Regresión Logística con TF-IDF y un modelo LSTM para capturar contexto, usando MLflow para rastreo.
Evaluación: La Regresión Logística logró un F1-score de 0.88, mientras que LSTM alcanzó 0.91.
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, TensorFlow (LSTM), NLTK, MLflow, WordCloud, Matplotlib.
Resultados: LSTM superó a la Regresión Logística al capturar mejor el contexto de las reseñas.Ver notebook | Ver gráficos


7. Modelado de Uplift
Construí un modelo de uplift para identificar clientes "persuadables" que responden positivamente a un tratamiento (publicidad) usando el dataset de Criteo (13M registros).

Carga y preprocesamiento: Cargué datos con Spark, resampling características numéricas.
Análisis exploratorio: Identifiqué patrones de visita y conversión, con un efecto del tratamiento de ~1% en visitas.
Modelado: Implementé un T-Learner con Logistic Regression (debido a problemas con LightGBM), usando MLflow para rastreo.
Evaluación: El modelo identificó el top 20% de persuadables, con un uplift predicho de 0.0022 a 0.0028.
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, MLflow, Matplotlib.
Resultados: El modelo es útil para optimizar campañas, pero el uplift es modesto debido a las tasas bajas de visita.Ver notebook | Ver gráficos


8. Detección de Fallos en Máquinas
Desarrollé un modelo de clasificación para detectar fallos en máquinas usando un dataset de mantenimiento predictivo con 10,000 registros.

Carga y limpieza: Cargué datos con Spark, escalé características numéricas.
Análisis exploratorio: Visualicé correlaciones y desbalance (0.17% fallos), usando SMOTETomek para balancear clases.
Modelado: Entrené Random Forest, Logistic Regression y XGBoost, usando MLflow para rastreo.
Evaluación: Random Forest logró un F1-score de 0.925 en prueba, con XGBoost alcanzando 0.973 pero con leve sobreajuste.
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Scikit-learn, XGBoost, Imbalanced-learn, MLflow, Seaborn, Matplotlib.
Resultados: Random Forest ofreció el mejor balance para mantenimiento predictivo.Ver notebook | Ver gráficos


9. Pronóstico de Ventas de Supermercado
Construí un modelo de pronóstico para predecir las ventas mensuales de la categoría "Furniture" usando el dataset de Superstore (9,995 registros).

Carga y preprocesamiento: Cargué datos con Spark, resampling a nivel mensual para la categoría "Furniture".
Análisis exploratorio: Visualicé estacionalidad y tendencia (2014-2017), descomponiendo la serie temporal.
Modelado: Entrené un modelo SARIMAX con order=(0, 1, 1) y seasonal_order=(0, 1, 1, 12), seleccionado por AIC (279.58).
Evaluación: MAPE de 15.24%, indicando buena precisión.
Pronósticos: Predije ventas para los próximos 6 meses (2023-2024).
Tecnologías: Python, Microsoft Fabric, Spark, Pandas, Statsmodels (SARIMAX), MLflow, Matplotlib.
Resultados: SARIMAX capturó bien los patrones estacionales, útil para planificar inventarios.Ver notebook | Ver gráficos

🛠️ Tecnologías Comunes

Entorno: Microsoft Fabric (Workspaces y Lakehouses específicos para cada ejercicio).
Librerías: PySpark, MLflow, Pandas, Scikit-learn, Imbalanced-learn, LightGBM, XGBoost, Statsmodels, Prophet, TensorFlow, NLTK, Surprise, Seaborn, Matplotlib, WordCloud.

🚀 ¿Cómo Navegar?
Cada carpeta de ejercicio contiene:

Un notebook con el código completo, explicaciones y resultados.
Datasets utilizados en formatos como CSV o Excel.
Gráficas generadas en PNG, como visualizaciones de datos y resultados de modelos.
Tablas Delta (si aplica).

Los notebooks son ejecutables en Microsoft Fabric o en entornos locales con Python y las librerías indicadas (ver requisitos en cada notebook).
📋 Requisitos

Microsoft Fabric (para ejecución nativa) o Python 3.11 con Jupyter.
Dependencias comunes: pandas, pyspark, scikit-learn, imbalanced-learn, lightgbm, xgboost, statsmodels, prophet, tensorflow, nltk, surprise, mlflow, seaborn, matplotlib, wordcloud (ver requirements.txt en cada ejercicio).

🌟 Reflexión
Este portafolio refleja mi capacidad para abordar problemas diversos de ciencia de datos, desde clasificación y pronósticos hasta sistemas de recomendación y detección de anomalías. Microsoft Fabric y MLflow me permitieron trabajar de manera eficiente con datos a gran escala, mientras que las técnicas avanzadas (como SMOTETomek, SARIMAX y SVD) mejoraron los resultados. Estoy emocionado de seguir explorando nuevas técnicas y aplicaciones en ciencia de datos.
📬 Contacto
👤 Juan Heriberto Rosas Juárez📧 Correo: juanheriberto.rosas@jhrjdata.com🌐 LinkedIn: Juan Heriberto Rosas Juárez🏢 Organización: Gobierno Digital e Innovación

