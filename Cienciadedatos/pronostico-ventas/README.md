Ejercicio 8: Pronóstico de Ventas de Supermercado 📈
En este proyecto, desarrollé un modelo de pronóstico para predecir las ventas mensuales de la categoría "Furniture" utilizando un dataset de Superstore con 9,995 registros. Implementé el flujo completo en Microsoft Fabric con Apache Spark y MLflow, utilizando el modelo SARIMAX para capturar patrones estacionales y tendencias en las ventas. A continuación, detallo el proceso técnico, los resultados obtenidos y las lecciones aprendidas.
🎯 Objetivo
El objetivo principal fue construir un modelo de pronóstico para predecir las ventas mensuales de la categoría "Furniture" en un supermercado, utilizando datos históricos de ventas (2014-2017). El modelo SARIMAX se empleó para capturar patrones estacionales y tendencias, con el fin de apoyar decisiones de inventario y planificación estratégica.
📊 Dataset
El dataset contiene 9,995 registros de ventas de productos en un supermercado, con 21 columnas iniciales que incluyen:

Order Date: Fecha del pedido.
Sales: Ventas en dólares.
Category: Categoría del producto (Furniture, Office Supplies, Technology).
Sub-Category, Product Name, Quantity, Discount, Profit, entre otras columnas de metadatos.

Desafíos:

Los datos están a nivel diario, pero el pronóstico requiere un resampling mensual.
Presencia de estacionalidad y tendencias que deben modelarse adecuadamente.

🛠️ Proceso Técnico
1. Carga y Preprocesamiento de Datos

Carga: Descargué el dataset (Superstore.xlsx) desde una URL pública y lo almacené en el lakehouse de Microsoft Fabric (saleslakehouse) en Files/salesforecast/raw/.
Preprocesamiento:
Filtré los datos para la categoría "Furniture".
Eliminé columnas innecesarias como Row ID, Order ID, Customer ID, etc., ya que el objetivo es pronosticar ventas totales por categoría.
Agrupé las ventas por fecha (Order Date) y las resampling a nivel mensual, calculando el promedio de ventas por mes.
Ajusté las fechas para simular un período más reciente (2023-2024) sumando 67 meses.



2. Análisis Exploratorio de Datos (EDA)

Rango de datos: Las ventas de Furniture abarcan desde enero de 2014 hasta diciembre de 2017.
Visualización:
Grafiqué las ventas mensuales, identificando picos estacionales anuales.
Descompuse la serie temporal en componentes: tendencia, estacionalidad y residuales. Esto reveló una clara estacionalidad de 12 meses y una tendencia general ascendente.



3. Modelado con SARIMAX

Ajuste de hiperparámetros:
Realicé una búsqueda de cuadrícula para los parámetros (p, d, q) y (P, D, Q, s) de SARIMAX, evaluando combinaciones con el criterio AIC.
La mejor combinación fue order=(0, 1, 1) y seasonal_order=(0, 1, 1, 12) con un AIC de 279.58.


Entrenamiento:
Entrené el modelo SARIMAX con los parámetros seleccionados, desactivando enforce_stationarity y enforce_invertibility para mayor flexibilidad.


Pronósticos:
Generé predicciones para los últimos 6 meses de datos observados y los próximos 6 meses (2023-2024), incluyendo intervalos de confianza.



4. Evaluación

Métrica: Calculé el MAPE (Mean Absolute Percentage Error) para los últimos 6 meses de datos observados:
MAPE: 15.24%, indicando buena precisión (un error promedio del 15.24% respecto a las ventas reales).


Almacenamiento: Combiné las ventas reales y pronosticadas, guardando los resultados como tabla Delta (Demand_Forecast_New_1) para visualización en Power BI.

📈 Resultados y Conclusiones

Precisión:
El MAPE de 15.24% indica que el modelo SARIMAX captura bien los patrones estacionales y de tendencia, con un error aceptable para aplicaciones prácticas.


Pronósticos:
Las predicciones para los próximos 6 meses (2023-2024) muestran continuidad en los patrones estacionales, útiles para planificar inventarios y estrategias de marketing.


Lecciones aprendidas:
La descomposición de la serie temporal fue clave para entender los componentes estacionales y de tendencia, guiando la selección del modelo.
SARIMAX es efectivo para series temporales estacionales, pero ajustar hiperparámetros con AIC mejora significativamente el rendimiento.
El MAPE podría reducirse con más datos o incluyendo variables exógenas (como descuentos o eventos promocionales).



🛠️ Tecnologías Utilizadas

Entorno: Microsoft Fabric (Workspace: pronostico-ventas, Lakehouse: saleslakehouse).
Librerías:
PySpark: Para procesamiento distribuido de datos.
MLflow: Para rastreo y registro de experimentos.
Statsmodels (SARIMAX): Para modelado de series temporales.
Pandas: Para manipulación de datos.
Matplotlib: Para visualización.



📂 Estructura del Repositorio
Ejercicio-8-Pronostico-Ventas/
├── sales_forecast.ipynb                         # Notebook con el código completo
├── data/
│   ├── Superstore.xlsx                          # Dataset original
│   ├── Demand_Forecast_New_1.csv                # Predicciones y datos combinados
├── results/
│   ├── sales_over_time.png                      # Ventas a lo largo del tiempo
│   ├── decomposition_plots.png                  # Descomposición de la serie temporal
│   ├── forecast_plot.png                        # Pronóstico con intervalos de confianza
├── README.md                                    # Este archivo

🚀 ¿Cómo Reproducir Este Proyecto?

Configura el entorno:
Crea una carpeta pronostico-ventas en Microsoft Fabric.
Añade un lakehouse (saleslakehouse).
Crea un notebook (sales_forecast.ipynb) y vincúlalo al lakehouse.


Ejecuta el notebook:
Sigue los bloques de código en orden (carga, EDA, modelado, evaluación, predicción).
Asegúrate de guardar las gráficas generadas.


Descarga los archivos:
Descarga el notebook, dataset, predicciones y gráficas siguiendo las instrucciones del bloque 5.


Explora los resultados:
Revisa las métricas (MAPE) y gráficas para entender el rendimiento del modelo.



🌟 Reflexión
Este proyecto fue una excelente oportunidad para trabajar con series temporales y aplicar SARIMAX en un contexto de pronóstico de ventas. La integración con Microsoft Fabric y MLflow facilitó el rastreo de experimentos y la visualización de resultados. En el futuro, me gustaría explorar variables exógenas (como promociones) o modelos más avanzados como Prophet para mejorar la precisión.
👤 Autor: Juan Heriberto Rosas Juárez📧 Correo: juanheriberto.rosas@jhrjdata.com🌐 LinkedIn: Juan Heriberto Rosas Juárez🏢 Organización: Gobierno Digital e Innovación
