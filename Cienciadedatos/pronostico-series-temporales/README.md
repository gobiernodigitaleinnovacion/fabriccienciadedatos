Ejercicio 4: Pronóstico de Series Temporales de Ventas de Propiedades 🏘️

En este proyecto, desarrollé un modelo de pronóstico de series temporales para predecir las ventas mensuales totales de propiedades residenciales en Nueva York (2003-2015). Implementé el flujo completo en Microsoft Fabric utilizando Spark y MLflow, empleando el modelo Prophet para capturar tendencias y estacionalidad, y comparando diferentes configuraciones de sensibilidad a cambios de tendencia. A continuación, detallo el proceso técnico, los resultados obtenidos y las lecciones aprendidas.

🎯 Objetivo
El objetivo principal fue construir un modelo de pronóstico para predecir las ventas mensuales de propiedades residenciales en Nueva York, identificando patrones estacionales y tendencias a largo plazo. Este modelo permite a los agentes inmobiliarios y planificadores urbanos anticipar fluctuaciones en el mercado, optimizando estrategias de inversión y gestión de inventario.

📊 Dataset
El dataset contiene registros de ventas de propiedades en Nueva York desde 2003 hasta 2015, extraído del archivo nyc_property_sales.tar. Incluye columnas como:

Fecha de venta: Fecha de la transacción.
Tipo de propiedad: Clasificación (residencial, comercial, etc.).
Precio de venta: Monto de la transacción.
Ubicación: Detalles geográficos (distrito, vecindario, etc.).

Estadísticas iniciales:

Período: 2003-2015 (12 años de datos mensuales).
Filtrado: Solo propiedades residenciales.

Desafíos:

Patrones estacionales complejos que requieren un modelo robusto para capturarlos.
Posibles cambios de tendencia debido a eventos económicos (por ejemplo, la crisis de 2008).

🛠️ Proceso Técnico
1. Carga y Limpieza de Datos

Carga:
Cargué el dataset (nyc_property_sales.tar) con Spark desde el lakehouse de Microsoft Fabric (timelakehouse).


Limpieza y preprocesamiento:
Filtré los datos para incluir solo propiedades residenciales.
Agregué las ventas a nivel mensual, sumando el número total de transacciones por mes.



2. Análisis Exploratorio de Datos (EDA)

Patrones estacionales:
Identifiqué picos de ventas en febrero y septiembre, y caídas en marzo y octubre.
Visualicé la serie temporal y sus componentes (tendencia, estacionalidad, residuales) mediante gráficos.


Tendencias:
Observé una tendencia general ascendente interrumpida por caídas significativas, probablemente relacionadas con eventos económicos como la crisis de 2008.



3. Modelado

Modelos entrenados:
Utilicé Prophet para modelar la serie temporal, configurando estacionalidad multiplicativa y MCMC para estimar incertidumbre.
Entrené tres modelos con diferentes valores de sensibilidad a cambios de tendencia:
changepoint_prior_scale = 0.01 (baja sensibilidad).
changepoint_prior_scale = 0.05 (sensibilidad moderada).
changepoint_prior_scale = 0.1 (alta sensibilidad).




Rastreo:
Usé MLflow para rastrear los experimentos, registrando métricas y parámetros de cada modelo.



4. Evaluación

Métricas:
Realicé validación cruzada para evaluar el desempeño de los modelos.
Mejor modelo (changepoint_prior_scale = 0.05):
MAPE (1 mes): ~8%.
MAPE (1 año): ~10%.


Los otros modelos mostraron mayor error:
changepoint_prior_scale = 0.01: MAPE ~12% (1 año), subestimando cambios de tendencia.
changepoint_prior_scale = 0.1: MAPE ~11% (1 año), sobreajustando a fluctuaciones menores.




Visualizaciones:
Grafiqué las predicciones junto con los datos reales, incluyendo intervalos de incertidumbre.



5. Predicciones

Generación:
Usé el mejor modelo (changepoint_prior_scale = 0.05) para generar predicciones por lotes.


Almacenamiento:
Guardé las predicciones en el lakehouse como tabla Delta para análisis futuro y visualización.



📈 Resultados y Conclusiones

Estacionalidad:
El modelo capturó bien los patrones estacionales:
Picos de ventas en febrero y septiembre.
Caídas en marzo y octubre.




Rendimiento:
El mejor modelo (changepoint_prior_scale = 0.05) logró un MAPE de ~8% para predicciones a 1 mes y ~10% para predicciones a 1 año, indicando buena precisión.


Insights:
Los picos estacionales sugieren que febrero y septiembre son períodos clave para la actividad inmobiliaria, probablemente debido a factores estacionales como el inicio de ciclos escolares o fiscales.
La sensibilidad moderada a cambios de tendencia (changepoint_prior_scale = 0.05) equilibra la captura de tendencias a largo plazo sin sobreajustar a fluctuaciones menores.


Lecciones aprendidas:
Prophet es una herramienta poderosa para modelar series temporales con estacionalidad, especialmente con configuraciones multiplicativas.
Ajustar changepoint_prior_scale es crucial para equilibrar la flexibilidad del modelo frente a cambios de tendencia.
La incertidumbre estimada mediante MCMC es útil para evaluar la confiabilidad de las predicciones, pero podría mejorarse aumentando el número de muestras MCMC o ajustando otros parámetros como seasonality_prior_scale.



🛠️ Tecnologías Utilizadas

Entorno: Microsoft Fabric (Workspace: series-temporales, Lakehouse: timelakehouse).
Librerías:
PySpark: Para procesamiento distribuido de datos.
MLflow: Para rastreo y registro de experimentos.
Prophet: Para modelado de series temporales.
Seaborn, Matplotlib: Para visualización.



📂 Estructura del Repositorio
Ejercicio-4-Pronostico-Series-Temporales/
├── time_series_forecast.ipynb                   # Notebook con el código completo
├── data/
│   ├── nyc_property_sales.tar                   # Dataset original
├── results/
│   ├── sales_over_time.png                      # Ventas mensuales a lo largo del tiempo
│   ├── decomposition_plots.png                  # Descomposición de la serie temporal
│   ├── forecast_comparison.png                  # Comparación de predicciones y datos reales
├── README.md                                    # Este archivo

🚀 ¿Cómo Reproducir Este Proyecto?

Configura el entorno:
Crea una carpeta series-temporales en Microsoft Fabric.
Añade un lakehouse (timelakehouse).
Crea un notebook (time_series_forecast.ipynb) y vincúlalo al lakehouse.


Ejecuta el notebook:
Sigue los bloques de código en orden (carga, EDA, modelado, evaluación, predicciones).
Asegúrate de guardar las gráficas generadas.


Descarga los archivos:
Descarga el notebook, dataset y gráficas siguiendo las instrucciones en el notebook.


Explora los resultados:
Revisa las métricas (MAPE) y gráficas para entender el rendimiento del modelo.



🌟 Reflexión
Este proyecto fue una valiosa oportunidad para trabajar con series temporales y aprender a modelar tendencias y estacionalidad en un contexto inmobiliario. Prophet demostró ser una herramienta efectiva y fácil de usar, especialmente para capturar patrones estacionales complejos. En el futuro, me gustaría incorporar variables exógenas (como tasas de interés o indicadores económicos) para mejorar la precisión de las predicciones.
Ver notebook | Ver gráficos
👤 Autor: Juan Heriberto Rosas Juárez📧 Correo: juanheriberto.rosas@jhrjdata.com🌐 LinkedIn: Juan Heriberto Rosas Juárez🏢 Organización: Gobierno Digital e Innovación
