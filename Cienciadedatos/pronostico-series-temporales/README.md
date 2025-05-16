Ejercicio 4: Pronóstico de Series Temporales

Desarrollé un modelo de pronóstico de series temporales para predecir las ventas mensuales totales de propiedades en Nueva York (2003-2015), implementado en Microsoft Fabric con Spark y MLflow. Utilicé Prophet para modelar tendencias y estacionalidad, comparando diferentes configuraciones de sensibilidad a cambios de tendencia.
Proceso

Carga y limpieza: Cargué el dataset (nyc_property_sales.tar) y lo agregué a nivel mensual, filtrando solo propiedades residenciales.  
Análisis exploratorio: Identifiqué patrones estacionales (picos en febrero y septiembre, caídas en marzo y octubre).  
Modelado: Entrené tres modelos Prophet con changepoint_prior_scale = [0.01, 0.05, 0.1], usando estacionalidad multiplicativa y MCMC para incertidumbre.  
Evaluación: Realicé validación cruzada, seleccionando el modelo con changepoint_prior_scale = 0.05 como el mejor (MAPE ~8%-10%).  
Predicciones: Generé predicciones por lotes y las guardé en el lakehouse.

Resultados

Estacionalidad: Picos de ventas en febrero y septiembre, caídas en marzo y octubre.  
Mejor modelo: changepoint_prior_scale = 0.05, con MAPE de ~8% (1 mes) y ~10% (1 año).  
Conclusión: El modelo captura bien las tendencias y estacionalidad, pero podría mejorarse ajustando parámetros como el número de muestras MCMC.

Tecnologías utilizadas

Python, Microsoft Fabric, Spark, MLflow, Prophet, Seaborn, Matplotlib.

Archivos disponibles

Notebook  
Gráficas

