Ejercicio 5: Clasificación de Texto para Determinar el Género de Libros 📚

En este proyecto, desarrollé un modelo de clasificación de texto para determinar el género de libros (ficción o no ficción) basado únicamente en sus títulos, utilizando datos de la British Library. Implementé el flujo de trabajo completo en Microsoft Fabric con Apache Spark y MLflow, empleando técnicas de procesamiento de lenguaje natural (NLP) como word2vec para la vectorización de texto y regresión logística para la clasificación. A continuación, detallo el proceso, los resultados y las lecciones aprendidas.
🎯 Objetivo
El objetivo principal fue construir un modelo de aprendizaje automático capaz de clasificar libros como ficción o no ficción basándose únicamente en sus títulos. Este proyecto forma parte de una serie de ejercicios de ciencia de datos realizados en Microsoft Fabric, demostrando un flujo de trabajo integral de ciencia de datos: desde la carga y exploración de datos hasta el entrenamiento, evaluación y predicción con un modelo.
📊 Dataset
El dataset proviene de la British Library y contiene metadatos de libros digitalizados, incluyendo sus títulos y etiquetas de género (Fiction o Non-fiction) asignadas manualmente. Algunas características clave del dataset:

Archivo: blbooksgenre.csv
Tamaño: Miles de registros con títulos y etiquetas de género.
Columnas relevantes:
Title: Título del libro (texto a clasificar).
annotator_genre: Etiqueta de género (Fiction o Non-fiction).


Desafíos:
Los títulos son cortos y a menudo contienen ruido (puntuación, palabras irrelevantes).
Desbalance inicial entre las clases, que fue corregido durante el preprocesamiento.



🛠️ Proceso Técnico
1. Carga y Limpieza de Datos

Carga: El dataset (blbooksgenre.csv) se descargó desde una URL pública y se almacenó en el lakehouse de Microsoft Fabric (textclassificationlakehouse) en la ruta Files/title-genre-classification/raw/.
Limpieza:
Se seleccionaron las columnas relevantes (Title y annotator_genre).
Se filtraron las etiquetas válidas (Fiction, Non-fiction) y se eliminaron duplicados basados en los títulos.
Se balancearon las clases usando ClassBalancer de SynapseML para mitigar el desbalance inicial (peso de Fiction ajustado a 2.956).



2. Análisis Exploratorio de Datos (EDA)

Tokenización y eliminación de stopwords: Utilicé Tokenizer y StopWordsRemover de PySpark ML para procesar los títulos, dividiéndolos en tokens y eliminando palabras comunes (stopwords) que no aportan valor semántico.
Visualización con nubes de palabras:
Generé nubes de palabras para cada clase usando la librería wordcloud:
Fiction: Palabras como "poem", "novel", "tale" predominan, reflejando narrativas y creatividad.
Non-fiction: Palabras como "history", "study", "bibliographical" son frecuentes, indicando temas históricos o académicos.


Estas visualizaciones se guardaron como wordcloud_fiction.png y wordcloud_nonfiction.png.



3. Vectorización del Texto

Word2Vec: Utilicé Word2Vec de PySpark ML para convertir los tokens filtrados en vectores de 128 dimensiones (word2vec_size=128). Se configuró un conteo mínimo de palabras (min_word_count=3) para ignorar términos poco frecuentes.
Transformación de etiquetas: Las etiquetas de género se convirtieron a índices numéricos usando StringIndexer (Fiction → 1.0, Non-fiction → 0.0).
Resultado: Un DataFrame con columnas Title, annotator_genre, features (vectores), labelIdx (etiquetas numéricas) y weight (peso de balanceo).

4. Entrenamiento del Modelo

División del dataset: Se dividió en 80% para entrenamiento y 20% para prueba (randomSplit([0.8, 0.2], seed=42)).
Modelo: Entrené un modelo de regresión logística (LogisticRegression) con las siguientes configuraciones:
maxIter=10: Máximo de 10 iteraciones.
Características: Vectores de word2vec (features).
Etiquetas: labelIdx.
Peso: weight (para balanceo de clases).


Optimización de hiperparámetros:
Utilicé validación cruzada (CrossValidator) con 3 pliegues (numFolds=3).
Cuadrícula de hiperparámetros: regParam=[0.03, 0.1], elasticNetParam=[0.0, 0.1].


Métricas de evaluación: areaUnderROC y areaUnderPR (evaluador binario, ya que hay dos clases).

5. Evaluación del Modelo

Resultados de los submodelos:
Submodelo 1 (regParam=0.03, elasticNetParam=0.0): areaUnderROC=0.7698, areaUnderPR=0.7115.
Submodelo 2 (regParam=0.03, elasticNetParam=0.1): areaUnderROC=0.7435.
Submodelo 3 (regParam=0.1, elasticNetParam=0.0): areaUnderROC=0.7565, areaUnderPR=0.6888.
Submodelo 4 (regParam=0.1, elasticNetParam=0.1): areaUnderROC=0.7337, areaUnderPR=0.6631.


Mejor modelo: El submodelo 1 (regParam=0.03, elasticNetParam=0.0) fue seleccionado por tener el mayor areaUnderROC (0.7698).

6. Predicciones y Almacenamiento

Predicciones por lotes: Usé el mejor modelo (versión 5 en MLflow) para generar predicciones en el conjunto de prueba.
Ejemplo de predicciones:
Título: 'Harpstrings.' A poem → Predicción: Fiction (correcto).
Título: A Balkán félsziget... → Predicción: Fiction (incorrecto, debería ser Non-fiction).


Almacenamiento: Las predicciones se guardaron en el lakehouse en Files/title-genre-classification/predictions/batch_predictions como archivos Delta.

7. Tiempo de ejecución

El notebook completo tomó 2085 segundos (~34 minutos), lo que incluye la instalación de librerías, preprocesamiento, entrenamiento, evaluación y predicción.

📈 Resultados y Conclusiones

Patrones identificados:
Las nubes de palabras revelaron diferencias claras entre géneros: los títulos de ficción suelen incluir términos narrativos, mientras que los de no ficción tienen un enfoque más académico o histórico.


Desempeño del modelo:
El mejor modelo logró un areaUnderROC de 0.7698 y un areaUnderPR de 0.7115, indicando un desempeño aceptable para un problema de clasificación binaria.
Sin embargo, el modelo comete errores (por ejemplo, clasificando algunos títulos de no ficción como ficción), lo que sugiere que los títulos cortos pueden no ser suficientes para una clasificación precisa.


Lecciones aprendidas:
La vectorización con word2vec captura bien el contexto semántico, pero el tamaño del vector (word2vec_size=128) y el conteo mínimo de palabras (min_word_count=3) podrían ajustarse para mejorar el modelo.
Los títulos cortos son un desafío para la clasificación; incorporar más contexto (como resúmenes o metadatos adicionales) podría mejorar los resultados.
El balanceo de clases fue crucial para mitigar el desbalance inicial, pero podría explorarse otras técnicas como SMOTE.



🛠️ Tecnologías Utilizadas

Entorno: Microsoft Fabric (Workspace: clasificacion-texto, Lakehouse: textclassificationlakehouse).
Librerías:
PySpark: Para procesamiento distribuido de datos.
SynapseML: Para balanceo de clases (ClassBalancer).
MLflow: Para rastreo de experimentos y registro de modelos.
Word2Vec y LogisticRegression: Para vectorización y clasificación (PySpark ML).
WordCloud, Seaborn, Matplotlib: Para visualización.
Python: Lenguaje principal.



📂 Estructura del Repositorio
Ejercicio-5-Clasificacion-Texto/
├── text_classification.ipynb    # Notebook con el código completo
├── data/
│   ├── blbooksgenre.csv         # Dataset original
│   ├── batch_predictions.csv    # Predicciones generadas
├── results/
│   ├── wordcloud_fiction.png    # Nube de palabras para Fiction
│   ├── wordcloud_nonfiction.png # Nube de palabras para Non-fiction
├── README.md                    # Este archivo

🚀 ¿Cómo Reproducir Este Proyecto?

Configura el entorno:
Crea una carpeta clasificacion-texto en Microsoft Fabric.
Añade un lakehouse (textclassificationlakehouse).
Crea un notebook (text_classification.ipynb) y vincúlalo al lakehouse.


Ejecuta el notebook:
Sigue los bloques de código en orden (instalación, carga de datos, EDA, entrenamiento, evaluación, predicciones).
Asegúrate de guardar las visualizaciones generadas (wordcloud_fiction.png, wordcloud_nonfiction.png).


Descarga los archivos:
Descarga el notebook, el dataset, las predicciones y las gráficas siguiendo las instrucciones del bloque 6.


Explora los resultados:
Revisa las métricas (areaUnderROC, areaUnderPR) y las nubes de palabras para entender los patrones en los datos.



🌟 Reflexión
Este proyecto fue una excelente oportunidad para explorar técnicas de NLP y clasificación en un entorno distribuido como Microsoft Fabric. Aunque el modelo mostró un desempeño prometedor, trabajar con títulos cortos destacó las limitaciones de este enfoque y la importancia de un buen preprocesamiento y selección de características. En el futuro, me gustaría experimentar con modelos más avanzados (como transformers) y datos adicionales para mejorar la precisión.
👤 Autor: Juan Heriberto Rosas Juárez📧 Correo: juanheriberto.rosas@jhrjdata.com🌐 LinkedIn: Juan Heriberto Rosas Juárez🏢 Organización: Gobierno Digital e Innovación
