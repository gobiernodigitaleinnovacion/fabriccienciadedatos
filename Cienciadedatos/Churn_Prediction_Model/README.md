Ejercicio 1: Predicción de Abandono de Clientes 🏦

En este proyecto, desarrollé un modelo de machine learning para predecir el abandono de clientes de un banco, utilizando un dataset con 10,000 registros. Implementé el flujo completo en Microsoft Fabric, abarcando desde la carga y limpieza de datos hasta el modelado, evaluación y análisis de resultados. A continuación, detallo el proceso técnico, los resultados obtenidos y las lecciones aprendidas.

🎯 Objetivo
El objetivo principal fue construir un modelo de clasificación para predecir si un cliente abandonará el banco (Exited=1) o no (Exited=0), basado en características demográficas y financieras. Este modelo permite a los bancos identificar clientes en riesgo y tomar medidas proactivas para retenerlos, optimizando estrategias de fidelización.

📊 Dataset
El dataset contiene 10,000 registros de clientes de un banco, con las siguientes columnas principales:

RowNumber, CustomerId, Surname: Identificadores del cliente (eliminados para el modelado).
CreditScore: Puntaje crediticio.
Geography: Ubicación (Francia, España, Alemania).
Gender: Género (Masculino, Femenino).
Age: Edad del cliente.
Tenure: Años como cliente.
Balance: Saldo de la cuenta.
NumOfProducts: Número de productos bancarios utilizados.
HasCrCard, IsActiveMember: Indicadores binarios de tarjeta de crédito y actividad.
EstimatedSalary: Salario estimado.
Exited: Etiqueta binaria (1 = abandono, 0 = no abandono).

Desafíos:

Desbalance de clases: Solo un 20% de los clientes abandonaron (Exited=1).
Variables categóricas (Geography, Gender) que requieren codificación.

🛠️ Proceso Técnico
1. Carga y Limpieza de Datos

Carga:
Cargué el dataset con Spark desde un archivo CSV almacenado en el lakehouse de Microsoft Fabric (churnlakehouse).


Limpieza:
Eliminé columnas irrelevantes (RowNumber, CustomerId, Surname) que no aportan valor predictivo.
Verifiqué y eliminé duplicados para garantizar la calidad de los datos.



2. Análisis Exploratorio de Datos (EDA)

Visualización:
Generé gráficos de barras e histogramas para explorar patrones de abandono:
Por geografía: Alemania tiene la mayor tasa de abandono (32.51%) frente a Francia (16.34%).
Por género: Las mujeres tienen una tasa de abandono del 25.07%, frente al 16.46% de los hombres.
Por edad: Los clientes mayores tienden a abandonar más.




Desbalance de clases:
Confirmé que Exited=1 representa solo el 20% de los datos, lo que requiere técnicas de balanceo.



3. Ingeniería de Características

Creación de variables:
NewTenure: Relación Tenure/Age para capturar la antigüedad relativa del cliente.


Discretización:
Discretizé variables numéricas como CreditScore, Age, Balance, y EstimatedSalary en rangos para capturar patrones no lineales.


Codificación:
Convertí variables categóricas (Geography, Gender) a variables dummy para el modelado.



4. Modelado

Modelos entrenados:
Entrené dos modelos Random Forest (max_depth=4 y max_depth=8) para establecer una línea base.
Entrené un modelo LightGBM con SMOTE para manejar el desbalance de clases.


Rastreo:
Usé MLflow para rastrear los experimentos, registrando métricas y parámetros de cada modelo.



5. Evaluación

Métricas:
Comparé los modelos usando matrices de confusión y métricas como precisión, recall y F1-score.
LightGBM tuvo el mejor rendimiento:
Precisión para la clase de abandono (Exited=1): 74%.
F1-score: 0.72 (promedio ponderado).


Random Forest (max_depth=8) alcanzó una precisión del 68% para la clase de abandono.


Análisis de tasas de abandono:
Por geografía: 32.51% en Alemania, 16.34% en Francia.
Por género: 25.07% mujeres, 16.46% hombres.
Por actividad: Los clientes inactivos (IsActiveMember=0) tienen una mayor probabilidad de abandono.



📈 Resultados y Conclusiones

Rendimiento:
LightGBM superó a Random Forest, logrando un equilibrio superior entre precisión y recall, con menos falsos positivos.
El uso de SMOTE mejoró significativamente la capacidad del modelo para identificar la clase minoritaria (Exited=1).


Insights:
Los clientes en Alemania y las mujeres tienen mayor probabilidad de abandono, lo que sugiere oportunidades para estrategias de retención específicas.
La edad y la inactividad son factores clave que influyen en el abandono.


Lecciones aprendidas:
El manejo del desbalance de clases es crucial para problemas de clasificación como este.
La ingeniería de características (como NewTenure) puede mejorar el rendimiento del modelo al capturar relaciones no lineales.
MLflow facilita la comparación y rastreo de experimentos en entornos distribuidos.



🛠️ Tecnologías Utilizadas

Entorno: Microsoft Fabric (Workspace: abandono-clientes, Lakehouse: churnlakehouse).
Librerías:
PySpark: Para procesamiento distribuido de datos.
MLflow: Para rastreo y registro de experimentos.
Pandas: Para manipulación de datos.
Scikit-learn: Para modelado (Random Forest).
LightGBM: Para modelado avanzado.
Imbalanced-learn (SMOTE): Para balanceo de clases.
Seaborn, Matplotlib: Para visualización.



📂 Estructura del Repositorio
Ejercicio-1-Abandono-de-clientes/
├── churn_prediction.ipynb                       # Notebook con el código completo
├── data/
│   ├── churn_data.csv                           # Dataset original
├── results/
│   ├── churn_by_geography.png                   # Tasa de abandono por geografía
│   ├── churn_by_gender.png                      # Tasa de abandono por género
│   ├── age_distribution.png                     # Distribución de edad
│   ├── confusion_matrix_lightgbm.png            # Matriz de confusión (LightGBM)
├── README.md                                    # Este archivo

🚀 ¿Cómo Reproducir Este Proyecto?

Configura el entorno:
Crea una carpeta abandono-clientes en Microsoft Fabric.
Añade un lakehouse (churnlakehouse).
Crea un notebook (churn_prediction.ipynb) y vincúlalo al lakehouse.


Ejecuta el notebook:
Sigue los bloques de código en orden (carga, EDA, ingeniería de características, modelado, evaluación).
Asegúrate de guardar las gráficas generadas.


Descarga los archivos:
Descarga el notebook, dataset y gráficas siguiendo las instrucciones en el notebook.


Explora los resultados:
Revisa las métricas (precisión, recall, F1-score) y gráficas para entender el rendimiento del modelo.



🌟 Reflexión
Este proyecto fue una excelente introducción al manejo de datos desbalanceados y a la construcción de modelos de clasificación en Microsoft Fabric. Aprendí la importancia de la ingeniería de características y el balanceo de clases, así como el valor de MLflow para gestionar experimentos. En el futuro, me gustaría explorar técnicas más avanzadas, como redes neuronales, para mejorar aún más la predicción de abandono.
Ver notebook | Ver gráficos
👤 Autor: Juan Heriberto Rosas Juárez📧 Correo: juanheriberto.rosas@jhrjdata.com🌐 LinkedIn: Juan Heriberto Rosas Juárez🏢 Organización: Gobierno Digital e Innovación
