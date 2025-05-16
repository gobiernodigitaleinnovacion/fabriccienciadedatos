Ejercicio 3: Detección de Fraudes

Desarrollé un modelo de detección de fraudes utilizando un dataset de transacciones de tarjetas de crédito de septiembre de 2013, implementado en Microsoft Fabric con Spark y MLflow. El objetivo fue identificar transacciones fraudulentas utilizando LightGBM, comparando el desempeño con datos desbalanceados y balanceados (usando SMOTE).  
Proceso

Carga y limpieza: Cargué el dataset (creditcard.csv) con 284,807 transacciones y 31 columnas (características V1-V28, Time, Amount, Class).  
Análisis exploratorio: Confirmé el desbalance extremo (99.83% no fraudes, 0.17% fraudes) mediante visualizaciones.  
Preparación de datos: Dividí el dataset en entrenamiento (85%) y prueba (15%), y apliqué SMOTE para balancear las clases en el conjunto de entrenamiento.  
Modelado: Entrené dos modelos LightGBM: uno con datos desbalanceados y otro con datos balanceados (SMOTE).  
Evaluación: Comparé el desempeño con métricas AUROC y AUPRC, y generé visualizaciones de importancia de características y matrices de confusión.  
Predicciones: Usé el mejor modelo (balanceado con SMOTE) para realizar predicciones por lotes y las guardé en el lakehouse.

Resultados

Distribución de clases: 99.83% no fraudes (284,315 transacciones), 0.17% fraudes (492 transacciones).  
Modelo con datos desbalanceados: AUROC: 0.7002, AUPRC: 0.0880.  
Modelo con datos balanceados (SMOTE): AUROC: 0.9253, AUPRC: 0.6410.  
Conclusión: El modelo balanceado con SMOTE mostró un desempeño mucho mejor (AUROC 0.9253 vs. 0.7002), destacando la efectividad de SMOTE para problemas desbalanceados. Sin embargo, hay espacio para mejoras, como ajustar hiperparámetros o probar otros algoritmos (por ejemplo, Random Forest o redes neuronales).

Tecnologías utilizadas

Python, Microsoft Fabric, Spark, MLflow, LightGBM, Scikit-learn, Seaborn, Matplotlib.

Archivos disponibles

Notebook  
Gráficas

