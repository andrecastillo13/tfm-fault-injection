# Fault Injection for TFLite CNNs (TFM)

Biblioteca **modular** para la inyección de fallos en modelos de redes neuronales convolucionales (CNNs) en **TensorFlow Lite (TFLite)**.  
El objetivo es proporcionar un **protocolo reproducible** que permita evaluar la **robustez frente a fallos hardware** en arquitecturas CNN bajo diferentes representaciones de precisión.

## ✨ Características principales

- Inyección de fallos a **nivel de bit** sobre tensores de pesos y sesgos.  
- Tres experimentos implementados:
  - **SBF (Single Bit Flip)** → detección de tensores sensibles a una única alteración.  
  - **Bit-Sweep** → caracterización de la sensibilidad según la posición del bit (LSB–MSB).  
  - **MBF (Multiple Bit Flips)** → estimación del umbral de colapso \(K\) para degradación crítica.  
- Ejecución **en memoria** sobre el *flatbuffer* de TFLite → sin necesidad de escribir ficheros intermedios.  
- Salidas reproducibles en **CSV / Pandas DataFrames**, listas para análisis posterior.  
- Modularidad: cada bloque corresponde a un script/cuaderno fácilmente reutilizable.  

## 🧪 Modelos soportados

La biblioteca se ha probado sobre las siguientes arquitecturas CNN exportadas a TFLite:

- **LeNet-5** → entrenado desde cero en **CIFAR-10**.  
- **MobileNetV2** → con subconjunto de **ImageNet-V2**.  
- **EfficientNet-B0** → con subconjunto de **ImageNet-V2**.  
- **GoogLeNet / InceptionV3** → con subconjunto de **ImageNet-V2**.  

Cada modelo se analiza en tres representaciones TFLite:  
- **FP32** (coma flotante 32 bits).  
- **CH** (cuantización híbrida).  
- **INT8** (cuantización completa post-training).  

## 📂 Estructura del repositorio

```bash
├── data/          # Conjuntos de datos utilizados en los experimentos
├── docs/          # Documentación del proyecto en LaTeX
├── models/        # Modelos entrenados/exportados a TFLite
├── notebooks/     # Cuadernos Colab (1 por arquitectura)
├── results/       # Resultados experimentales (CSV, gráficos, métricas)
├── src/           # Código fuente modular (scripts de inyección y utilidades)
│   ├── load_imagenet_v2_test.py
│   ├── preprocess_input_tflite.py
│   ├── exportar_tflite_variantes.py
│   ├── evaluate_model.py
│   ├── inspeccionar_tensores_inyectables.py
│   ├── calcular_muestreo_estadistico.py
│   ├── campaña_sbf_modelo.py
│   ├── bit_sweep_modelo.py
│   ├── campaña_mbf_modelo.py
│   └── clasificar_tensores_sensibles.py
├── LICENSE.txt
├── README.md
└── requirements.txt
```
⚙️ Instalación y dependencias
Clonar el repositorio:
```bash
git clone https://github.com/andrecastillo13/tfm-fault-injection.git
cd tfm-fault-injection
```
Instalar dependencias (entorno local):
```bash
pip install -r requirements.txt
```
Dependencias principales:

- TensorFlow / TFLite (TF 2.x)
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Flatbuffers

En Google Colab Pro, solo es necesario instalar Flatbuffers manualmente:
```bash
!pip install flatbuffers
```
---

## 🚀 Uso básico

Ejemplo de flujo de uso para un modelo TFLite:

1. **Exportar variantes** (FP32, CH, INT8):  
   ```bash
   python src/exportar_tflite_variantes.py
   ```

2. **Evaluar modelo y baseline**:  
   ```bash
   python src/evaluate_model.py
   ```

3. **Ejecutar campañas de inyección**:  
   - Single Bit Flip (SBF)  
     ```bash
     python src/campaña_sbf_modelo.py
     ```
   - Bit-Sweep  
     ```bash
     python src/bit_sweep_modelo.py
     ```
   - Multiple Bit Flips (MBF)  
     ```bash
     python src/campaña_mbf_modelo.py
     ```

📂 Los resultados de cada campaña se almacenan en la carpeta `results/` en formato **CSV** y con visualizaciones gráficas para su análisis.

---

## 🔧 Entorno de experimentación

- **Google Colab Pro** (51 GB RAM, 225 GB almacenamiento)  
- **Python 3.10**  
- **TensorFlow/TFLite 2.x**, **NumPy**, **Pandas**, **Matplotlib**, **Flatbuffers**, **scikit-learn**, **seaborn**

---

## 📊 Ejemplo de workflow completo

```bash
# 1. Exportar variantes del modelo (ej. LeNet-5)
python src/exportar_tflite_variantes.py

# 2. Evaluar baseline en el dataset de prueba
python src/evaluate_model.py

# 3. Detectar tensores sensibles (SBF)
python src/campaña_sbf_modelo.py

# 4. Ejecutar barrido de posiciones de bit (Bit-Sweep)
python src/bit_sweep_modelo.py

# 5. Estimar umbral de colapso con múltiples flips (MBF)
python src/campaña_mbf_modelo.py
```

---

## 📜 Licencia

Este proyecto se distribuye bajo licencia **MIT**, lo que permite su uso, modificación y distribución libre, siempre que se mantenga la atribución correspondiente.


