Ejercicio 2: Sistema de Recomendación de Libros (Análisis Comparativo)

En este proyecto, desarrollé un sistema de recomendación de libros basado en las preferencias de los usuarios, utilizando un modelo de filtrado colaborativo con ALS (Alternating Least Squares) en Microsoft Fabric. Realicé dos experimentos: un experimento inicial y un experimento mejorado para intentar optimizar el rendimiento. A continuación, detallo el proceso, los resultados y las lecciones aprendidas.
Experimento Inicial
Proceso

Carga y limpieza: Cargué los datos con Spark desde el Book-Crossing Dataset (Books.csv, Ratings.csv, Users.csv), eliminando duplicados y valores nulos.  
Visualización: Generé gráficos exploratorios para identificar patrones, como los autores con más libros y los libros más populares.  
Ingeniería de características: Transformé User-ID e ISBN en índices enteros (_user_id, _item_id) para el modelo ALS.  
Modelado: Entrené un modelo ALS con hiperparámetros (rank=64, regParam=[0.01, 0.1], maxIter=1), usando MLflow para rastreo. Usé una muestra de 5,000 filas para acelerar el entrenamiento.  
Evaluación: Evalué el modelo con métricas RMSE, MAE, R2 y varianza explicada.  
Análisis: Generé las top 10 recomendaciones de libros para cada usuario.

Resultados

Total de usuarios: 278,858 usuarios distintos.
Total de libros: 271,360 libros distintos.
Total de interacciones usuario-libro: 1,031,136 interacciones.
Dispersión del dataset: 99.7465% (muy alta).
Rendimiento del modelo ALS (mejor submodelo):
RMSE: 4.7458 (error promedio alto).
MAE: 4.4646 (error absoluto promedio también alto).
R2: -14.9927 (negativo, indica que el modelo no explica los datos).
Varianza explicada: 21.1474.


Top libros más populares: "Wild Animus" (2,502 calificaciones), "The Lovely Bones: A Novel" (1,295), "The Da Vinci Code" (883).
Top recomendaciones generadas: "Lasher: Lives of the Mayfair Witches", "The Da Vinci Code", entre otros.

Experimento Mejorado
Proceso
Para abordar las limitaciones del experimento inicial, realicé ajustes:  

Carga y limpieza: Filtré usuarios y libros con menos de 5 interacciones para reducir la dispersión.  
Datos utilizados: Usé el dataset completo (sin muestreo) para tener más datos.  
Modelado: Ajusté los hiperparámetros (rank=[16, 32], regParam=[0.01, 0.1], maxIter=5) y usé MLflow para rastreo.  
Evaluación y análisis: Generé nuevas recomendaciones y comparé métricas con el experimento inicial.

Resultados

Total de usuarios: 22,072 usuarios (filtrados para mayor calidad).
Total de libros: 40,909 libros (filtrados para mayor calidad).
Total de interacciones usuario-libro: 600,148 interacciones (después del filtrado).
Dispersión del dataset: 99.9326% (ligeramente reducida).
Rendimiento del modelo ALS (mejor submodelo):
RMSE: 2.5447 (mejorado respecto al experimento inicial: 4.7458).
MAE: 1.9916 (mejorado respecto al experimento inicial: 4.4646).
R2: -1.0012 (negativo, pero mejorado respecto al experimento inicial: -14.9927).
Varianza explicada: 5.0675 (menor que el experimento inicial: 21.1474).


Top libros más populares: "Wild Animus" (1,686 calificaciones), "The Lovely Bones: A Novel" (981), "The Da Vinci Code" (722).
Top recomendaciones generadas: "The Elementary Particles", "The Pugilist at Rest", "The Da Vinci Code", entre otros.

Análisis: ¿Por qué los modelos no son adecuados?
A pesar de las mejoras en el segundo experimento, ambos modelos tienen un R² negativo, lo que indica que no explican bien los datos y son peores que una predicción simple basada en la media de las calificaciones. Las razones principales son:  

Alta dispersión del dataset: Incluso después del filtrado, la dispersión sigue siendo muy alta (99.9326%), lo que dificulta que el modelo ALS encuentre patrones significativos.  
Limitaciones del modelo ALS: ALS puede no ser el mejor enfoque para datasets tan dispersos. Modelos alternativos, como redes neuronales o enfoques basados en contenido, podrían ser más efectivos.  
Interacciones implícitas no aprovechadas: En el experimento mejorado, usé implicitPrefs=False para depuración, lo que limitó la capacidad del modelo para manejar calificaciones implícitas (rating=0).  
Hiperparámetros y datos: Aunque ajusté los hiperparámetros, las configuraciones probadas podrían no ser óptimas, y el volumen de datos sigue siendo insuficiente para un dataset tan disperso.

Valor del proyecto a pesar de los resultados
Aunque los modelos no son adecuados para predicciones precisas debido a su R² negativo, este proyecto tiene valor como ejercicio de aprendizaje:  

Exploración y limpieza de datos: Aprendí a manejar datasets dispersos, filtrar datos irrelevantes y reducir la dispersión para mejorar la calidad de los datos.  
Visualización y análisis: Los gráficos generados (como los libros más populares) son útiles para entender patrones en los datos, incluso sin un modelo efectivo.  
Generación de recomendaciones: A pesar de las métricas pobres, las recomendaciones generadas (como "The Da Vinci Code") pueden ser útiles para usuarios en un contexto exploratorio, ya que identifican libros populares y relevantes.  
Uso de herramientas modernas: Gané experiencia con Microsoft Fabric, Spark, MLflow, Pandas, Seaborn y Matplotlib, herramientas clave en ciencia de datos.  
Lecciones aprendidas: Entendí las limitaciones de ALS en datasets dispersos y la importancia de manejar datos implícitos, lo que me ayudará a diseñar mejores sistemas en el futuro.

Tecnologías utilizadas

Python, Microsoft Fabric, Spark, MLflow, Pandas, Seaborn, Matplotlib.

Archivos disponibles

Experimento Inicial:Ver notebook | Ver gráficos  
Experimento Mejorado:Ver notebook | Ver gráficos

