# Carpeta `models/`

En esta carpeta se almacenan **los modelos TFLite exportados** a partir de las arquitecturas evaluadas en el TFM:

- **LeNet-5** (entrenado desde cero en CIFAR-10).  
- **MobileNetV2**, **GoogLeNet/InceptionV3** y **EfficientNet-B0** (convertidos desde modelos preentrenados en ImageNet-V2).

Cada arquitectura se genera en **tres variantes**:
- `FP32` – Exportación directa a TFLite en punto flotante.  
- `CH` – Cuantización híbrida (pesos en INT8, operaciones en FP32).  
- `INT8` – Cuantización completa con *Post-Training Quantization (PTQ)* y *representative dataset*.  

---

## Cómo generar los modelos

La exportación se realiza mediante el script:

```bash
python src/exportar_tflite_variantes.py
```

Este script toma como entrada el modelo entrenado en TensorFlow/Keras y produce automáticamente las tres variantes anteriores, que se guardan en esta carpeta.  

Ejemplo de ejecución para **LeNet-5** entrenado en CIFAR-10:

```bash
python src/exportar_tflite_variantes.py --model lenet5 --dataset cifar10 --output models/
```

---

## Estructura esperada

```
models/
├── lenet5_fp32.tflite
├── lenet5_ch.tflite
├── lenet5_int8.tflite
├── mobilenetv2_fp32.tflite
├── mobilenetv2_ch.tflite
├── mobilenetv2_int8.tflite
...
```

---

## Notas importantes

- Los modelos **no se versionan en GitHub** para mantener el repositorio ligero.  
- Cada usuario debe generarlos localmente con los scripts provistos.  
- En caso de necesitar un modelo de ejemplo, puede usarse un archivo reducido en la carpeta `samples/`.

---
