Ejercicio 2: Sistema de Recomendación de Libros (Análisis Comparativo) 📚

En este proyecto, desarrollé un sistema de recomendación de libros basado en las preferencias de los usuarios, utilizando un modelo de filtrado colaborativo con ALS (Alternating Least Squares) en Microsoft Fabric. Realicé dos experimentos: un experimento inicial para establecer una línea base y un experimento mejorado para optimizar el rendimiento. A continuación, detallo el proceso técnico, los resultados, un análisis crítico de los modelos y las lecciones aprendidas.

🎯 Objetivo
El objetivo principal fue construir un sistema de recomendación que sugiera libros a los usuarios basándose en sus calificaciones históricas, utilizando el dataset Book-Crossing. Comparé dos enfoques para evaluar cómo las mejoras en el preprocesamiento y los hiperparámetros afectan el rendimiento del modelo ALS, con el fin de generar recomendaciones más precisas y útiles.

📊 Dataset
El dataset Book-Crossing contiene tres archivos principales:

Books.csv: Información de libros (ISBN, título, autor, etc.).
Ratings.csv: Calificaciones de usuarios (User-ID, ISBN, rating de 0 a 10).
Users.csv: Datos de usuarios (User-ID, ubicación, edad).

Estadísticas iniciales:

Usuarios: 278,858 usuarios distintos.
Libros: 271,360 libros distintos.
Interacciones usuario-libro: 1,031,136 calificaciones.

Desafíos:

Alta dispersión del dataset (99.7465%), lo que dificulta encontrar patrones significativos.
Presencia de calificaciones implícitas (rating=0) que requieren un manejo especial.

🛠️ Experimentos
Experimento Inicial
Proceso

Carga y limpieza:
Cargué los datos con Spark desde Books.csv, Ratings.csv, y Users.csv.
Eliminé duplicados y valores nulos para garantizar la calidad de los datos.
Usé una muestra de 5,000 filas para acelerar el entrenamiento.


Visualización:
Generé gráficos exploratorios para identificar patrones:
Autores con más libros publicados.
Libros más populares según las calificaciones.




Ingeniería de características:
Transformé User-ID e ISBN en índices enteros (_user_id, _item_id) para el modelo ALS.


Modelado:
Entrené un modelo ALS con hiperparámetros: rank=64, regParam=[0.01, 0.1], maxIter=1.
Usé MLflow para rastrear los experimentos.


Evaluación:
Calculé métricas: RMSE, MAE, R² y varianza explicada.


Análisis:
Generé las top 10 recomendaciones de libros para cada usuario.



Resultados

Estadísticas:
Total de usuarios: 278,858.
Total de libros: 271,360.
Total de interacciones usuario-libro: 1,031,136.
Dispersión del dataset: 99.7465% (muy alta).


Rendimiento del modelo ALS (mejor submodelo):
RMSE: 4.7458 (error promedio alto).
MAE: 4.4646 (error absoluto promedio también alto).
R²: -14.9927 (negativo, indica que el modelo no explica los datos).
Varianza explicada: 21.1474.


Top libros más populares:
"Wild Animus" (2,502 calificaciones).
"The Lovely Bones: A Novel" (1,295 calificaciones).
"The Da Vinci Code" (883 calificaciones).


Top recomendaciones generadas:
"Lasher: Lives of the Mayfair Witches".
"The Da Vinci Code".



Análisis inicial
El modelo inicial mostró un rendimiento pobre (R² negativo), principalmente debido a la alta dispersión del dataset y la muestra limitada de datos. Esto motivó un segundo experimento para abordar estas limitaciones.

Experimento Mejorado
Proceso
Para mejorar los resultados, realicé los siguientes ajustes:

Carga y limpieza:
Filtré usuarios y libros con menos de 5 interacciones para reducir la dispersión y mejorar la calidad de los datos.
Usé el dataset completo (sin muestreo) para aprovechar más interacciones.


Modelado:
Ajusté los hiperparámetros: rank=[16, 32], regParam=[0.01, 0.1], maxIter=5.
Continué usando MLflow para rastrear los experimentos.


Evaluación y análisis:
Generé nuevas recomendaciones y comparé métricas con el experimento inicial.



Resultados

Estadísticas después del filtrado:
Total de usuarios: 22,072 (filtrados para mayor calidad).
Total de libros: 40,909 (filtrados para mayor calidad).
Total de interacciones usuario-libro: 600,148.
Dispersión del dataset: 99.9326% (ligeramente reducida).


Rendimiento del modelo ALS (mejor submodelo):
RMSE: 2.5447 (mejorado respecto al experimento inicial: 4.7458).
MAE: 1.9916 (mejorado respecto al experimento inicial: 4.4646).
R²: -1.0012 (negativo, pero mejorado respecto al experimento inicial: -14.9927).
Varianza explicada: 5.0675 (menor que el experimento inicial: 21.1474).


Top libros más populares:
"Wild Animus" (1,686 calificaciones).
"The Lovely Bones: A Novel" (981 calificaciones).
"The Da Vinci Code" (722 calificaciones).


Top recomendaciones generadas:
"The Elementary Particles".
"The Pugilist at Rest".
"The Da Vinci Code".




📉 Análisis Crítico: ¿Por qué los modelos no son adecuados?
A pesar de las mejoras en el segundo experimento, ambos modelos tienen un R² negativo, lo que indica que no explican bien los datos y son peores que una predicción simple basada en la media de las calificaciones. Las razones principales son:

Alta dispersión del dataset:
Incluso después del filtrado, la dispersión sigue siendo muy alta (99.9326%), lo que dificulta que ALS encuentre patrones significativos.


Limitaciones del modelo ALS:
ALS puede no ser el mejor enfoque para datasets tan dispersos. Modelos alternativos, como redes neuronales (por ejemplo, embeddings) o enfoques basados en contenido (usando metadatos de libros), podrían ser más efectivos.


Interacciones implícitas no aprovechadas:
En el experimento mejorado, usé implicitPrefs=False para depuración, lo que limitó la capacidad del modelo para manejar calificaciones implícitas (rating=0). Configurar implicitPrefs=True podría haber mejorado los resultados.


Hiperparámetros y datos:
Aunque ajusté los hiperparámetros, las configuraciones probadas (rank, regParam, maxIter) podrían no ser óptimas. Además, el volumen de datos sigue siendo insuficiente para un dataset tan disperso.



🌟 Valor del Proyecto
Aunque los modelos no son adecuados para predicciones precisas debido a su R² negativo, este proyecto tiene un valor significativo como ejercicio de aprendizaje:

Exploración y limpieza de datos:
Aprendí a manejar datasets dispersos, filtrar datos irrelevantes y reducir la dispersión para mejorar la calidad de los datos.


Visualización y análisis:
Los gráficos generados (como los libros más populares) son útiles para entender patrones en los datos, incluso sin un modelo efectivo.


Generación de recomendaciones:
A pesar de las métricas pobres, las recomendaciones generadas (como "The Da Vinci Code") identifican libros populares y relevantes, lo que puede ser útil en un contexto exploratorio para usuarios.


Uso de herramientas modernas:
Gané experiencia con Microsoft Fabric, Spark, MLflow, Pandas, Seaborn y Matplotlib, herramientas clave en ciencia de datos.


Lecciones aprendidas:
Entendí las limitaciones de ALS en datasets dispersos y la importancia de manejar datos implícitos, lo que me ayudará a diseñar mejores sistemas de recomendación en el futuro.



🛠️ Tecnologías Utilizadas

Entorno: Microsoft Fabric (Workspace: recomendaciones-libros, Lakehouse: booklakehouse).
Librerías:
PySpark: Para procesamiento distribuido de datos.
MLflow: Para rastreo y registro de experimentos.
Pandas: Para manipulación de datos.
Seaborn, Matplotlib: Para visualización.



📂 Estructura del Repositorio
Ejercicio-2-Recomendaciones/
├── recommendation_system.ipynb                   # Notebook del experimento inicial
├── recommendation_system_improved.ipynb          # Notebook del experimento mejorado
├── data/
│   ├── Books.csv                                 # Datos de libros
│   ├── Ratings.csv                               # Calificaciones de usuarios
│   ├── Users.csv                                 # Datos de usuarios
├── results/
│   ├── initial_popular_books.png                 # Gráfica de libros populares (experimento inicial)
│   ├── initial_author_distribution.png           # Distribución de autores (experimento inicial)
│   ├── improved_popular_books.png                # Gráfica de libros populares (experimento mejorado)
│   ├── improved_author_distribution.png          # Distribución de autores (experimento mejorado)
├── README.md                                     # Este archivo

🚀 ¿Cómo Reproducir Este Proyecto?

Configura el entorno:
Crea una carpeta recomendaciones-libros en Microsoft Fabric.
Añade un lakehouse (booklakehouse).
Crea dos notebooks (recommendation_system.ipynb y recommendation_system_improved.ipynb) y vincúlalos al lakehouse.


Ejecuta los notebooks:
Sigue los bloques de código en cada notebook (carga, EDA, modelado, evaluación).
Asegúrate de guardar las gráficas generadas.


Descarga los archivos:
Descarga los notebooks, datasets y gráficas siguiendo las instrucciones en cada notebook.


Explora los resultados:
Revisa las métricas (RMSE, MAE, R²) y las gráficas para entender el rendimiento de los modelos.



🌟 Reflexión
Este proyecto fue una valiosa oportunidad para explorar los desafíos de los sistemas de recomendación con datasets dispersos. Aunque ALS no fue adecuado para este caso, el proceso me permitió aprender técnicas de preprocesamiento, manejo de dispersión y evaluación de modelos. En el futuro, planeo explorar modelos híbridos (filtrado colaborativo + contenido) o enfoques basados en redes neuronales para mejorar las recomendaciones.
Ver notebook del Experimento Inicial | Ver gráficos del Experimento InicialVer notebook del Experimento Mejorado | Ver gráficos del Experimento Mejorado
👤 Autor: Juan Heriberto Rosas Juárez📧 Correo: juanheriberto.rosas@jhrjdata.com🌐 LinkedIn: Juan Heriberto Rosas Juárez🏢 Organización: Gobierno Digital e Innovación

