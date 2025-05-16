Ejercicio 6: Modelado de Uplift para Identificar Persuadables 📈
En este proyecto, desarrollé un modelo de uplift utilizando un T-Learner para identificar usuarios "persuadables" que responden positivamente a un tratamiento (publicidad), empleando el dataset de Criteo. Implementé el flujo de trabajo en Microsoft Fabric con Apache Spark y MLflow, inicialmente intentando usar LightGBMClassifier, pero debido a problemas de compatibilidad, opté por LogisticRegression de PySpark ML. A continuación, detallo el proceso técnico, los resultados obtenidos y las lecciones aprendidas.
🎯 Objetivo
El objetivo principal fue construir un modelo de uplift para estimar el efecto incremental de un tratamiento (publicidad) en el comportamiento de los usuarios, específicamente en su probabilidad de visitar una tienda online. El modelo clasifica a los usuarios en grupos como "persuadables" (aquellos que responden positivamente al tratamiento), optimizando así las estrategias de marketing. Este proyecto forma parte de una serie de ejercicios de ciencia de datos realizados en Microsoft Fabric, demostrando un flujo completo de modelado de uplift.
📊 Dataset
El dataset utilizado proviene de Criteo AI Lab y contiene 13M de registros, cada uno representando a un usuario. Incluye las siguientes columnas:

f0 a f11: 12 características numéricas (valores flotantes densos) que describen a los usuarios.
treatment: Indicador binario (1 = usuario expuesto al tratamiento, 0 = control).
visit: Etiqueta binaria (1 = usuario visitó la tienda online, 0 = no visitó).
conversion: Etiqueta binaria (1 = usuario realizó una compra, 0 = no compró).
exposure: Indicador de exposición (no utilizado en este análisis).

Citación requerida:
@inproceedings{Diemert2018,
  author = {{Diemert Eustache, Betlei Artem} and Renaudin, Christophe and Massih-Reza, Amini},
  title={A Large Scale Benchmark for Uplift Modeling},
  publisher = {ACM},
  booktitle = {Proceedings of the AdKDD and TargetAd Workshop, KDD, London, United Kingdom, August, 20, 2018},
  year = {2018}
}

Desafíos:

Desbalance entre los grupos de tratamiento (85% tratamiento, ~15% control) y tasas bajas de visita (4.7%) y conversión (~0.29%).
Problemas de compatibilidad con LightGBMClassifier en SynapseML, lo que llevó al uso de LogisticRegression.

🛠️ Proceso Técnico
1. Carga y Preprocesamiento de Datos

Carga: Descargué el dataset (criteo-research-uplift-v2.1.csv, 13M filas) desde una URL pública y lo almacené en el lakehouse de Microsoft Fabric (upliftlakehouse) en la ruta Files/uplift-modelling/raw/.
Preprocesamiento:
Convertí las columnas de características (f0 a f11) a tipo double para asegurar consistencia numérica.
Utilicé VectorAssembler de PySpark ML para combinar las columnas f0 a f11 en una única columna de características features (vector).



2. Análisis Exploratorio de Datos (EDA)

Estadísticas generales:
~4.7% de los usuarios visitaron la tienda online.
~0.29% de los usuarios convirtieron (realizaron una compra).
~6.2% de los visitantes convirtieron.


Estadísticas por grupo de tratamiento:
Tratamiento (treatment=1): 4.85% visitaron, 0.31% convirtieron, 6.36% de los visitantes convirtieron.
Control (treatment=0): 3.82% visitaron, 0.19% convirtieron, 5.07% de los visitantes convirtieron.
Efecto del tratamiento: Mejora la tasa de visitas en ~1% y la conversión de visitantes en ~1.3%.



3. Preparación de Datos para el Modelo

Vectorización: Usé VectorAssembler para crear la columna features a partir de f0 a f11.
División del dataset: Dividí el dataset en entrenamiento (80%, ~11.18M filas) y prueba (20%, ~2.80M filas).
Separación de grupos:
Tratamiento (treatment_train_df): 9.5M filas (85%).
Control (control_train_df): 1.7M filas (15%).



4. Entrenamiento del Modelo T-Learner

Modelo: Implementé un T-Learner con LogisticRegression de PySpark ML (debido a problemas con LightGBMClassifier):
Configuración: maxIter=100, regParam=0.01, elasticNetParam=0.0 (regularización L2).
Entrené dos modelos:
Modelo de tratamiento: Sobre treatment_train_df (~9.5M filas).
Modelo de control: Sobre control_train_df (~1.7M filas).




Registro: Los modelos se registraron en MLflow bajo los nombres aisample-upliftmodelling-treatmentmodel y aisample-upliftmodelling-controlmodel.

5. Predicción y Cálculo del Uplift

Predicciones: Usé ambos modelos para predecir la probabilidad de visita (visit) en el conjunto de prueba.
Uplift predicho: Calculé pred_uplift como la diferencia entre la probabilidad predicha por el modelo de tratamiento (treatment_pred) y el modelo de control (control_pred).
Resultados:
Valores de pred_uplift entre 0.0022 y 0.0028, indicando un efecto modesto del tratamiento.



6. Evaluación y Curva de Uplift

Curva de uplift:
Ordené los usuarios por pred_uplift (descendente) y calculé el uplift acumulado (group_uplift) para diferentes proporciones de la población.
Visualicé la curva de uplift, mostrando que el top 20% de la población (persuadables) maximiza el uplift.
La gráfica se guardó como UpliftCurve.png en MLflow.


Punto de corte: Identifiqué un cutoff_score para el top 20%, registrado en MLflow.

📈 Resultados y Conclusiones

Uplift predicho:
Los valores de pred_uplift oscilan entre 0.0022 y 0.0028, reflejando un efecto modesto del tratamiento, consistente con el análisis exploratorio (~1% de mejora en visitas).


Curva de uplift:
El top 20% de la población (ordenada por uplift predicho) maximiza el uplift, identificando a los "persuadables".


Lecciones aprendidas:
Compatibilidad: Problemas con LightGBMClassifier en SynapseML (duplicidad de nombres de columnas internas) me llevaron a usar LogisticRegression. Esto sugiere la necesidad de verificar la compatibilidad de las bibliotecas en entornos distribuidos como Fabric.
Uplift modesto: El uplift predicho es pequeño, lo que podría deberse a las tasas bajas de visita en el dataset. Modelos más complejos (como LightGBM, si se resuelven los problemas) o características adicionales podrían mejorar los resultados.
Escalabilidad: El procesamiento de 13M de registros en Spark fue eficiente, pero el desbalance entre los grupos de tratamiento y control (~85%/15%) requiere un manejo cuidadoso.



🛠️ Tecnologías Utilizadas

Entorno: Microsoft Fabric (Workspace: modelado-uplift, Lakehouse: upliftlakehouse).
Librerías:
PySpark: Para procesamiento distribuido de datos.
MLflow: Para rastreo y registro de experimentos.
LogisticRegression (PySpark ML): Para modelado.
VectorAssembler (PySpark ML): Para preprocesamiento de características.
Matplotlib: Para visualización de la curva de uplift.
Python: Lenguaje principal.



📂 Estructura del Repositorio
Ejercicio-6-Modelado-Uplift/
├── uplift_modeling.ipynb                # Notebook con el código completo
├── data/
│   ├── criteo-research-uplift-v2.1.csv  # Dataset original
│   ├── batch_predictions_treatment_control.csv  # Predicciones generadas
├── results/
│   ├── UpliftCurve.png                  # Curva de uplift
├── README.md                            # Este archivo

🚀 ¿Cómo Reproducir Este Proyecto?

Configura el entorno:
Crea una carpeta modelado-uplift en Microsoft Fabric.
Añade un lakehouse (upliftlakehouse).
Crea un notebook (uplift_modeling.ipynb) y vincúlalo al lakehouse.


Ejecuta el notebook:
Sigue los bloques de código en orden (carga, EDA, preprocesamiento, entrenamiento, predicción, evaluación).
Asegúrate de guardar la gráfica generada (UpliftCurve.png).


Descarga los archivos:
Descarga el notebook, el dataset, las predicciones y la gráfica siguiendo las instrucciones del bloque 7.


Explora los resultados:
Revisa la curva de uplift y el cutoff_score para identificar persuadables.



🌟 Reflexión
Este proyecto fue una excelente oportunidad para explorar el modelado de uplift y trabajar con datasets grandes en un entorno distribuido como Microsoft Fabric. A pesar de los desafíos con LightGBMClassifier, el uso de LogisticRegression permitió completar el análisis, aunque con un uplift predicho modesto. En el futuro, me gustaría abordar los problemas de compatibilidad con LightGBM o explorar otros meta-learners (como S-Learner o X-Learner) para mejorar las predicciones.
👤 Autor: Juan Heriberto Rosas Juárez📧 Correo: juanheriberto.rosas@jhrjdata.com🌐 LinkedIn: Juan Heriberto Rosas Juárez🏢 Organización: Gobierno Digital e Innovación
