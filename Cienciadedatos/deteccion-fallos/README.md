Ejercicio 7: Detección de Fallos en Máquinas 🔧

En este proyecto, desarrollé un modelo de clasificación para predecir fallos en máquinas utilizando un dataset de mantenimiento predictivo con 10,000 registros. Implementé el flujo completo en Microsoft Fabric con Apache Spark y MLflow, entrenando y comparando tres modelos: Random Forest, Logistic Regression y XGBoost. A continuación, detallo el proceso técnico, los resultados obtenidos y las lecciones aprendidas.

🎯 Objetivo

El objetivo principal fue construir un modelo de clasificación para predecir si una máquina experimentará un fallo (IsFail=1) basado en características como temperatura del aire, temperatura del proceso, velocidad de rotación, torque y desgaste de la herramienta. Este proyecto forma parte de una serie de ejercicios de ciencia de datos realizados en Microsoft Fabric, demostrando un flujo completo de aprendizaje automático para mantenimiento predictivo.

📊 Dataset

El dataset simula el registro de parámetros de una máquina de fabricación en función del tiempo, común en entornos industriales. Contiene 10,000 registros con las siguientes columnas:





Type: Tipo de producto (L, M, H), indicando la variante de calidad (baja, media, alta).



Air_temperature_[K]: Temperatura del aire en Kelvin.



Process_temperature_[K]: Temperatura del proceso en Kelvin.



Rotational_speed_[rpm]: Velocidad de rotación en RPM.



Torque_[Nm]: Torque en Nm.



Tool_wear_[min]: Desgaste de la herramienta en minutos.



IsFail: Etiqueta binaria (1 = fallo, 0 = no fallo).



Failure_Type: Tipo de fallo (No Failure, TWF, HDF, PWF, OSF, RNF).

Desafíos:





Fuerte desbalance de clases: la mayoría de las máquinas no fallan (IsFail=0).



Múltiples modos de fallo, aunque el modelo solo predice IsFail (no el tipo de fallo).

🛠️ Proceso Técnico

1. Carga y Preprocesamiento de Datos





Carga: Descargué el dataset (predictive_maintenance.csv, 10,000 filas) desde una URL pública y lo almacené en el lakehouse de Microsoft Fabric (faultlakehouse) en Files/predictive_maintenance/raw/.



Preprocesamiento:





Reemplacé espacios en los nombres de las columnas por guiones bajos para evitar problemas en Spark.



Guardé el DataFrame como tabla Delta en Tables/predictive_maintenance_data.



Convertí el DataFrame a Pandas, eliminé columnas innecesarias (UDI, Product_ID), renombré Target a IsFail, y mapeé Type (L=0, M=1, H=2).

2. Análisis Exploratorio de Datos (EDA)





Matriz de correlación:





Alta correlación de IsFail con Rotational_speed_[rpm], Torque_[Nm], y Tool_wear_[min], lo que sugiere que estas características son clave para predecir fallos.



Distribuciones:





Las características numéricas (Air_temperature_[K], Process_temperature_[K], etc.) no son dispersas y tienen buena continuidad, adecuadas para modelado.



Desbalance de clases:





La mayoría de las muestras son IsFail=0 (sin fallo), con una minoría significativa de IsFail=1.



La columna Failure_Type mostró predominio de "No Failure", con pocos casos de otros tipos de fallo (TWF, HDF, PWF, OSF, RNF).

3. Preparación de Datos





Selección de características: Usé Type, Air_temperature_[K], Process_temperature_[K], Rotational_speed_[rpm], Torque_[Nm], y Tool_wear_[min].



División del dataset: 80% entrenamiento (8,000 filas), 20% prueba (2,000 filas).



Balanceo de clases: Apliqué SMOTETomek para balancear IsFail=0 y IsFail=1 en el conjunto de entrenamiento.

4. Entrenamiento y Evaluación de Modelos





Modelos entrenados:





Random Forest Classifier (max_depth=5, n_estimators=50).



Logistic Regression Classifier (random_state=42).



XGBoost Classifier (parámetros predeterminados).



Métricas:





Random Forest: F1-score prueba 0.925, accuracy 0.8955 (buen balance).



Logistic Regression: F1-score prueba 0.8869, accuracy 0.835 (menor rendimiento).



XGBoost: F1-score prueba 0.9728, accuracy 0.97 (mejor, pero con leve sobreajuste: F1-score entrenamiento 0.9975).



Registro: Todos los modelos y métricas se registraron en MLflow bajo el experimento Machine_Failure_Classification.

5. Predicción





Modelo seleccionado: Random Forest, por su balance entre rendimiento y generalización.



Predicciones: Usé Fabric PREDICT (MLFlowTransformer) para predecir fallos en el conjunto de prueba.



Almacenamiento: Guardé las predicciones en Tables/predictive_maintenance_test_with_predictions.

📈 Resultados y Conclusiones





Rendimiento:





Random Forest ofrece un buen equilibrio (F1-score prueba: 0.925), adecuado para mantenimiento predictivo.



XGBoost tiene el mejor rendimiento en prueba (F1-score: 0.9728), pero muestra leve sobreajuste.



Logistic Regression tiene el menor rendimiento (F1-score prueba: 0.8869).



Lecciones aprendidas:





El balanceo de clases con SMOTETomek fue crucial para mejorar la capacidad del modelo de detectar fallos (IsFail=1).



Características como Torque_[Nm] y Tool_wear_[min] son predictoras clave, según el análisis de correlación.



XGBoost podría beneficiarse de ajustes de hiperparámetros para reducir el sobreajuste.

🛠️ Tecnologías Utilizadas





Entorno: Microsoft Fabric (Workspace: deteccion-fallos, Lakehouse: faultlakehouse).



Librerías:





PySpark: Para procesamiento distribuido de datos.



MLflow: Para rastreo y registro de experimentos.



Scikit-learn: Para Random Forest y Logistic Regression.



XGBoost: Para el modelo XGBoost.



SMOTETomek (imblearn): Para balanceo de clases.



Seaborn, Matplotlib: Para visualización.

📂 Estructura del Repositorio

Ejercicio-7-Deteccion-Fallos/
├── fault_detection.ipynb                        # Notebook con el código completo
├── data/
│   ├── predictive_maintenance.csv               # Dataset original
│   ├── predictive_maintenance_test_data.csv     # Conjunto de prueba
│   ├── predictive_maintenance_test_with_predictions.csv  # Predicciones
├── results/
│   ├── correlation_heatmap.png                  # Matriz de correlación
│   ├── feature_histograms.png                   # Histogramas de características
│   ├── failure_type_counts.png                  # Conteo de tipos de fallo
│   ├── class_balance_counts.png                 # Conteo de clases (desbalanceado)
│   ├── balanced_class_counts.png                # Conteo de clases (balanceado)
├── README.md                                    # Este archivo

🚀 ¿Cómo Reproducir Este Proyecto?





Configura el entorno:





Crea una carpeta deteccion-fallos en Microsoft Fabric.



Añade un lakehouse (faultlakehouse).



Crea un notebook (fault_detection.ipynb) y vincúlalo al lakehouse.



Ejecuta el notebook:





Sigue los bloques de código en orden (carga, EDA, preprocesamiento, entrenamiento, predicción).



Asegúrate de guardar las gráficas generadas.



Descarga los archivos:





Descarga el notebook, dataset, predicciones y gráficas siguiendo las instrucciones del bloque 7.



Explora los resultados:





Revisa las métricas y gráficas para entender el rendimiento del modelo.

🌟 Reflexión

Este proyecto fue una gran oportunidad para trabajar con datos desbalanceados y aplicar técnicas de aprendizaje automático en un escenario de mantenimiento predictivo. El uso de SMOTETomek mejoró significativamente la capacidad del modelo para detectar fallos, y Random Forest demostró ser una opción robusta. En el futuro, me gustaría explorar ajustes de hiperparámetros para XGBoost y agregar más características al dataset para mejorar aún más el rendimiento.

👤 Autor: Juan Heriberto Rosas Juárez
📧 Correo: juanheriberto.rosas@jhrjdata.com
🌐 LinkedIn: Juan Heriberto Rosas Juárez
🏢 Organización: Gobierno Digital e Innovación
