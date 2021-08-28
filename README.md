# Trabajo Fin de Máster

En este repositorio puede consultarse el código desarrollado para el Trabajo Fin de Máster DATCOM, el cual es una actualización del [Trabajo Fin de Grado](https://github.com/laurahernandezm/TFG). Se trata de un sistema completo de detección y reconocimiento facial de individuos con trayectorias anómalas. Para probar el sistema completo, se montó un sistema CCTV formado por tres cámaras con puntos de vista diferentes y se ha creado un conjunto de vídeos privado. 

El sistema implementado toma como entrada una serie de vídeos (vista cenital) de los que se extraen las trayectorias con el método de tracking [FairMOT](https://github.com/ifzhang/FairMOT). A continuación, se aplica clustering sobre ellas para extraer el comportamiento normal y detectar, posteriormente, aquellas que sean anómalas en cuanto a dirección o velocidad. Aparte de la cámara cenital con la que se localiza a los individuos con trayectorias anormales, contamos con dos cámaras más, con puntos de vista diferentes, para reidentificar a estas personas dentro del Circuito Cerrado de Televisión y realizar el reconocimiento facial que confirme su identidad.

Dado que el conjunto de datos utilizado es para uso privado y exclusivo de los miembros del proyecto, los resultados obtenidos a partir de este y reflejados en la memoria del trabajo no son reproducibles. Con el código y los datos presentes en este repositorio se pueden obtener los resultados relativos a la detección de anomalías en trayectorias sobre el dataset [Peds1](https://drive.google.com/file/d/1l1XBHSr_XLlmGJRs_UrvZ0ExcGDDjzKI/view?usp=sharing).

## Estructura de la carpeta Anomaly_detections_Peds1

### gt

Este directorio contiene las ground truth actualizadas de cada vídeo del conjunto de test para realizar el cálculo de las métricas.

### FairMOT

En este directorio encontramos la implementación del algoritmo de tracking [FairMOT](https://github.com/ifzhang/FairMOT) con un archivo extra (src/demo_from_script.py) que hemos usado para automatizar el procesado de todos los vídeos que necesitamos y el archivo src/demo.py modificado.

### anomaly_detections.py

Este archivo contiene las funciones que se encargan de detectar las trayectorias anormales y dibujar los cuadros delimitadores correspondientes en los vídeos del dataset.

### create_test_folder.py

Este script construye el directorio con los datos de las trayectorias anómalas detectadas por el algoritmo para compararlos con las ground truth.

### create_tracking_results.py

Este script construye el directorio con las trayectorias obtenidas por el algoritmo de tracking.

### draw_gt_dets.py

Script para dibujar las detecciones del algoritmo (color rojo) y ground truth (color verde) sobre el mismo vídeo.

### empty_scene.jpg

Imagen que se utiliza como lienzo para dibujar la información relacionada con las zonas y la cuadrícula.

### metrics.py

Este script se encarga de computar las métricas tomando como entradas el directorio _gt_ y el directorio creado por _create_test_folder.py_.

### run.py

Script de ejecución completa (entrenamiento y test).

### test.py

Script para ejecutar únicamente la parte de test (es necesario haber ejecutado run.py o train.py previamente, al menos una vez, ya que se utiliza información generada durante el entrenamiento del sistema).

### test_functions.py

Este archivo contiene funciones que únicamente se ejecutan con los vídeos de test.

### train.py

Script para entrenar el modelo y obtener la información de las zonas necesaria para detectar las trayectorias anómalas.

### train_functions.py

Este archivo contiene funciones que se ejecutan durante el entrenamiento, para realizar el descubrimiento de zonas.

### utils.py

Este archivo contiene diversas funciones auxiliares, así como variables y constantes globales.

### zones.py

Este archivo contiene funciones que se encargan de actualizar la información sobre las zonas descubiertas durante el entrenamiento.

## Instrucciones para la reproducción de resultados en Anomaly_detections_Peds1

Para generar los mismos resultados mostrados en el trabajo (relativos al dataset Peds1), los pasos a seguir son los siguientes:
