# Ble-Analysis

Repositorio donde se almacena todo el trabajo de análisis, interpretación y transformación de los datos BLE recibidos en
hasta conseguir un Set de datos final, entrenamiento de algoritmos de Machine Learning y su posteior evaluación.

## Estructura del repositorio

Existen 6 archivos principales agrupados en la carpeta *src*:

- **funtions_off.py**: Archivo que contiene las funciones de preprocesamiento de los datos. Desde la carga de los datos
  en
  bruto hasta la obtención de los datos finales.
- **select_algorithm_lazy.py**: Archivo que usando la librería *LazyPredict*, realiza una comparativa de los algoritmos
  de
  Machine Learning más comunes para seleccionar el mejor.
- **train_models.py**: Archivo que contiene la función de entrenamiento de los modelos de Machine Learning, calculando
  los
  mejores hiperparámetros para el algoritmo.
- **test_models.py**: Archivo que contiene la función de evaluación de los modelos de Machine Learning, calculando las
  métricas de evaluación.
- **manipulation_training_evaluation.py**: Archivo que ejecuta las funciones en los archivos anteriores para la
  manipulación
  de los datos de entrenamiento y evaluación, entrenamiento de los algoritmos de Machine Learning y evaluación de los
  modelos entrenados.
- **cross_validation_graph.py**: Archivo que usando los algoritmos con sus hiperparámetros óptimos hace uso de la
  validación
  cruzada, devolviendo las predicciones calculadas en cada fold y graficando dichas predicciones para cada día.

## Ejecución del código

Los scripts funcionan correctamente con la versión 3.9 de Python. La lista con los paquetes y sus versiones instalados
están
detallados en el archivo **environment.yml**. Este archivo se puede usar para crear un entorno virtual con todos los
paquetes necesarios para ejecutar el código. Simplemente con este comando:

```bash
conda env create -f environment.yml
```

Los archivos csv con los datos deben estar divididos en tres carpetas y ordenados por fecha dentro de una carpeta
general llamada *data*, una para los datos BLE, otra para el contador de personas y una última sobre el estado de las
Raspberry Pi. Además, existen otras tres carpetas con la misma función albergando los datos usados para test. Estas
carpetas son recibididas como argumentos al comienzo del script **manipulation_training_evaluation.py**

Una vez el entorno está listo, se debe ejecutar el script **manipulation_training_evaluation.py** para obtener los
modelos entrenados y guardados en la carpeta que se le indique como argumento.
Este script genera gráficas y las guarda en la ruta *figures* con la etiqueta indicada como argumento de las
características calculadas sin procesar, las características seleccionadas y procesadas y las predicciones y el error de
los modelos entrenados.

Por último, se debe ejecutar el script **cross_validation_graph.py** para obtener las predicciones de los modelos
entrenados en cada fold y graficarlas.