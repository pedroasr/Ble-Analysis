# Ble-Analysis

Repositorio donde se almacena todo el trabajo de análisis, interpretación y transformación de los datos BLE recibidos en
hasta conseguir un Set de datos final, entrenamiento de algoritmos de Machine Learning y su posteior evaluación.

## Estructura del repositorio

Existen 7 archivos principales agrupados en la carpeta *src*:

- **clean_BLE_data.py**: Archivo que contiene la función que limpia los datos de los archivos ble brutos y guarda los
  datos en archivos .csv
- **funtions_off.py**: Archivo que contiene las funciones de preprocesamiento de los datos. Desde la carga de los datos
  en bruto hasta la obtención de los datos finales.
- **select_algorithm_lazy.py**: Archivo que usando la librería *LazyPredict*, realiza una comparativa de los algoritmos
  de Machine Learning más comunes para seleccionar el mejor.
- **train_models.py**: Archivo que contiene la función de entrenamiento de los modelos de Machine Learning, calculando
  los mejores hiperparámetros para el algoritmo.
- **test_models.py**: Archivo que contiene la función de evaluación de los modelos de Machine Learning, calculando las
  métricas de evaluación.
- **manipulation_training_evaluation.py**: Archivo que ejecuta las funciones en los archivos anteriores para la
  manipulación de los datos de entrenamiento y evaluación, entrenamiento de los algoritmos de Machine Learning y
  evaluación de los modelos entrenados.
- **cross_validation_graph.py**: Archivo que usando los algoritmos con sus hiperparámetros óptimos hace uso de la
  validación cruzada, devolviendo las predicciones calculadas en cada fold y graficando dichas predicciones para cada
  día.

## Ejecución del código

Los scripts funcionan correctamente con la versión 3.9 de Python. La lista con los paquetes y sus versiones instalados
están detallados en el archivo **ocupacion.yml** en la raíz del repositorio. Este archivo se puede usar para crear un
entorno virtual con todos los paquetes necesarios para ejecutar el código. Simplemente con este comando:

```bash
conda env create -f ocupacion.yml
```

Los archivos csv con los datos deben estar divididos en tres carpetas y ordenados por fecha dentro de una carpeta
general llamada *data*, una para los datos BLE, otra para el contador de personas y una última sobre el estado de las
Raspberry Pi, que servirán como datos de aprendizaje. Además, existen otras tres carpetas con la misma función
albergando los datos usados para la evaluación. Estas carpetas son recibidas como argumentos al comienzo del script
**manipulation_training_evaluation.py**. Todos los archivos almacenados en estas carpetas deben pertenecer a los mismos
días que se quieren procesar, de otro modo el script dará error. En el script **manipulation_training_evaluation.py** se
deben indicar la ruta de todas las carpetas y archivos necesarios para su ejecución. Una vez configurado esto, el script
se ejecuta de la siguiente manera:

```bash
python3 manipulation_training_evaluation.py
```

El script **manipulation_training_evaluation.py** limpia los datos en bruto y obtiene los modelos entrenados y los
guarda en la carpeta que se le indique como argumento, por defecto será *models*. Existe una variable llamada *sampling*
que indica el intervalo de tiempo en el que se van a agrupar los datos. Los datos procesados se guardan en la carpeta
*results*, en el directorio con la etiqueta pasada como argumento. Este script genera gráficas y las guarda en la ruta
*figures* con la etiqueta indicada como argumento de las características calculadas sin procesar, las características
seleccionadas y procesadas y las predicciones y el error de los modelos entrenados.

Para obtener los mejores algoritmos para los datos procesados usando la librería *LazyPredict*, se debe ejecutar el
script **select_algorithm_lazy.py**.

Por último, se debe ejecutar el script **cross_validation_graph.py** para obtener las predicciones de los modelos
entrenados en cada fold y graficarlas.