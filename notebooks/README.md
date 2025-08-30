# Carpeta `notebooks/`

En esta carpeta se incluyen los **cuadernos Jupyter/Colab** que documentan el flujo de trabajo de la inyección de fallos en distintos modelos.  
Cada cuaderno corresponde a una arquitectura de red neuronal y contiene las funciones modulares necesarias para ejecutar los experimentos.

---

## Contenido

- **01_LeNet5_Final.ipynb**  
  Entrenamiento de LeNet-5 desde cero en CIFAR-10, exportación a TFLite (FP32, CH, INT8) y ejecución de campañas SBF, Bit-Sweep y MBF.  
  Sirve como **modelo base** y caso didáctico.

- **02_MobileNetV2_Final.ipynb**  
  Evaluación de MobileNetV2 sobre un subconjunto de ImageNet-V2. Se realizan las tres campañas de inyección y se comparan con LeNet-5.

- **03_GoogLeNet_InceptionV3_Final.ipynb**  
  Análisis de GoogLeNet/InceptionV3 en TFLite. Incluye pruebas de sensibilidad de tensores y estudio del umbral de colapso K.

- **04_EfficientNetB0_Final.ipynb**  
  Evaluación de EfficientNet-B0 con variantes TFLite. Se aplican los experimentos completos (SBF, Bit-Sweep, MBF) y se registran métricas.

---

## Recomendaciones de uso

1. Abrir los cuadernos en **Google Colab Pro** (se recomienda por memoria y estabilidad).  
2. Ejecutar las celdas en orden; cada cuaderno está diseñado como un flujo autónomo.  
3. Modificar parámetros globales (semilla, subconjunto, progresión K) en las primeras celdas de configuración.  
4. Los resultados se almacenan automáticamente en la carpeta `results/` con subcarpetas por modelo.

---

## Notas

- Los cuadernos están pensados para **documentar** y **reproducir** los experimentos descritos en la memoria del TFM.  
- Para una ejecución modular fuera de Colab, se recomienda usar directamente los scripts en la carpeta `src/`.  
- Se sugiere limpiar los **outputs pesados** antes de versionar, de manera que el repositorio se mantenga ligero.

---
