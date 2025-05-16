Ejercicio 3: Detección de Fraudes en Transacciones 💳
En este proyecto, desarrollé un modelo de detección de fraudes para identificar transacciones fraudulentas en un dataset de transacciones con tarjetas de crédito de septiembre de 2013. Implementé el flujo completo en Microsoft Fabric utilizando Spark y MLflow, comparando el desempeño de un modelo LightGBM con datos desbalanceados y balanceados (usando SMOTE). A continuación, detallo el proceso técnico, los resultados obtenidos y las lecciones aprendidas.
🎯 Objetivo
El objetivo principal fue construir un modelo de clasificación para detectar transacciones fraudulentas (Class=1) frente a transacciones legítimas (Class=0), utilizando características anonimizadas derivadas de PCA. Este modelo permite a las instituciones financieras identificar fraudes de manera proactiva, reduciendo pérdidas y mejorando la seguridad de las transacciones.
📊 Dataset
El dataset contiene 284,807 transacciones realizadas con tarjetas de crédito en septiembre de 2013, con las siguientes columnas:

Time: Segundos transcurridos desde la primera transacción.
Amount: Monto de la transacción.
V1 a V28: Características anonimizadas (resultados de PCA para proteger la privacidad).
Class: Etiqueta binaria (1 = fraude, 0 = no fraude).

Estadísticas iniciales:

Distribución de clases:
No fraudes: 99.83% (284,315 transacciones).
Fraudes: 0.17% (492 transacciones).



Desafíos:

Desbalance extremo (0.17% de fraudes), lo que requiere técnicas de balanceo para evitar sesgos hacia la clase mayoritaria.
Características anonimizadas, lo que limita la interpretabilidad directa.

🛠️ Proceso Técnico
1. Carga y Limpieza de Datos

Carga:
Cargué el dataset (creditcard.csv) con Spark desde el lakehouse de Microsoft Fabric (fraudlakehouse).


Limpieza:
Verifiqué la ausencia de valores nulos y duplicados, asegurando la calidad de los datos.



2. Análisis Exploratorio de Datos (EDA)

Distribución de clases:
Confirmé el desbalance extremo: 99.83% no fraudes vs. 0.17% fraudes.
Visualicé la distribución mediante gráficos de barras.


Exploración de características:
Analicé las distribuciones de Time y Amount, y las correlaciones entre las características V1 a V28 (limitadas por la anonimización).



3. Preparación de Datos

División del dataset:
Dividí los datos en entrenamiento (85%, 242,086 transacciones) y prueba (15%, 42,721 transacciones).


Balanceo de clases:
Apliqué SMOTE (Synthetic Minority Oversampling Technique) al conjunto de entrenamiento para generar muestras sintéticas de la clase minoritaria (Class=1), reduciendo el desbalance.



4. Modelado

Modelos entrenados:
Modelo con datos desbalanceados: Entrené un modelo LightGBM sin balanceo para establecer una línea base.
Modelo con datos balanceados: Entrené un segundo modelo LightGBM usando los datos balanceados con SMOTE.


Rastreo:
Usé MLflow para rastrear los experimentos, registrando métricas y parámetros de cada modelo.



5. Evaluación

Métricas:
Evalué ambos modelos usando AUROC (Área bajo la curva ROC) y AUPRC (Área bajo la curva Precision-Recall), métricas adecuadas para problemas desbalanceados.
Modelo con datos desbalanceados:
AUROC: 0.7002.
AUPRC: 0.0880.


Modelo con datos balanceados (SMOTE):
AUROC: 0.9253.
AUPRC: 0.6410.




Visualizaciones:
Generé matrices de confusión para analizar falsos positivos y negativos.
Grafiqué la importancia de características, destacando que V14, V4, y V12 tienen un impacto significativo en la detección de fraudes.



6. Predicciones

Modelo seleccionado:
Utilicé el modelo balanceado con SMOTE (mejor desempeño: AUROC 0.9253) para realizar predicciones por lotes.


Almacenamiento:
Guardé las predicciones en el lakehouse como tabla Delta para análisis futuro.



📈 Resultados y Conclusiones

Rendimiento:
El modelo balanceado con SMOTE mostró una mejora significativa:
AUROC: 0.9253 vs. 0.7002 (modelo desbalanceado).
AUPRC: 0.6410 vs. 0.0880 (modelo desbalanceado).


Esto destaca la efectividad de SMOTE para problemas desbalanceados, mejorando la capacidad del modelo para detectar fraudes.


Insights:
Las características V14, V4, y V12 son las más influyentes para identificar fraudes, lo que sugiere patrones específicos en las transacciones fraudulentas.


Lecciones aprendidas:
El balanceo de clases es crucial para problemas con desbalance extremo, como la detección de fraudes.
AUPRC es una métrica más informativa que AUROC en datasets desbalanceados, ya que se enfoca en la clase minoritaria.
Hay espacio para mejoras: ajustar hiperparámetros de LightGBM o probar otros algoritmos como Random Forest o redes neuronales podría aumentar aún más la precisión.



🛠️ Tecnologías Utilizadas

Entorno: Microsoft Fabric (Workspace: deteccion-fraudes, Lakehouse: fraudlakehouse).
Librerías:
PySpark: Para procesamiento distribuido de datos.
MLflow: Para rastreo y registro de experimentos.
LightGBM: Para modelado.
Scikit-learn: Para métricas de evaluación.
Imbalanced-learn (SMOTE): Para balanceo de clases.
Seaborn, Matplotlib: Para visualización.



📂 Estructura del Repositorio
Ejercicio-3-Deteccion-de-fraudes/
├── fraud_detection.ipynb                        # Notebook con el código completo
├── data/
│   ├── creditcard.csv                           # Dataset original
├── results/
│   ├── class_distribution.png                   # Distribución de clases
│   ├── confusion_matrix_unbalanced.png          # Matriz de confusión (modelo desbalanceado)
│   ├── confusion_matrix_balanced.png            # Matriz de confusión (modelo balanceado)
│   ├── feature_importance.png                   # Importancia de características
├── README.md                                    # Este archivo

🚀 ¿Cómo Reproducir Este Proyecto?

Configura el entorno:
Crea una carpeta deteccion-fraudes en Microsoft Fabric.
Añade un lakehouse (fraudlakehouse).
Crea un notebook (fraud_detection.ipynb) y vincúlalo al lakehouse.


Ejecuta el notebook:
Sigue los bloques de código en orden (carga, EDA, preparación, modelado, evaluación, predicciones).
Asegúrate de guardar las gráficas generadas.


Descarga los archivos:
Descarga el notebook, dataset y gráficas siguiendo las instrucciones en el notebook.


Explora los resultados:
Revisa las métricas (AUROC, AUPRC) y gráficas para entender el rendimiento del modelo.



🌟 Reflexión
Este proyecto fue una oportunidad clave para trabajar con datasets desbalanceados y aplicar técnicas de detección de anomalías en un contexto financiero. Aprendí la importancia de métricas como AUPRC para problemas desbalanceados y el impacto del balanceo de clases con SMOTE. En el futuro, me gustaría explorar modelos más avanzados, como redes neuronales, o incorporar datos temporales adicionales para mejorar la detección de fraudes.
Ver notebook | Ver gráficos
👤 Autor: Juan Heriberto Rosas Juárez📧 Correo: juanheriberto.rosas@jhrjdata.com🌐 LinkedIn: Juan Heriberto Rosas Juárez🏢 Organización: Gobierno Digital e Innovación

