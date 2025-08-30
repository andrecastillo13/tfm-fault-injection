import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import mobilenet_v2, inception_v3, efficientnet

def exportar_tflite_variantes(modelo, x_sample, nombre_base="modelo_exportado"):
    """
    Exporta un modelo Keras a tres variantes TFLite: FP32, INT8 y CH (cuantización híbrida).

    Parámetros:
        modelo (tf.keras.Model): modelo Keras ya cargado.
        x_sample (np.array): muestra de entrada para la calibración (shape = [N, H, W, C]).
        nombre_base (str): nombre base para los archivos .tflite exportados.
    """

    # ----- Detectar tamaño de entrada -----
    input_shape = modelo.input_shape
    if isinstance(input_shape, list):  # Modelos con múltiples entradas
        input_shape = input_shape[0]
    target_size = input_shape[1:3]  # (H, W)

    # ----- Detectar preprocesado correcto según modelo -----
    preprocess_fn = lambda x: x  # por defecto, identidad
    if "mobilenet" in nombre_base.lower():
        preprocess_fn = mobilenet_v2.preprocess_input
    elif "inception" in nombre_base.lower() or "googlenet" in nombre_base.lower():
        preprocess_fn = inception_v3.preprocess_input
    elif "efficientnet" in nombre_base.lower():
        preprocess_fn = efficientnet.preprocess_input
    else:
        preprocess_fn = lambda x: x / 255.0

    # ----- Aplicar preprocesado a la muestra para calibrar INT8 -----
    x_sample_resized = tf.image.resize(x_sample, target_size)
    x_sample_resized = tf.cast(x_sample_resized, tf.float32)
    x_sample_resized = preprocess_fn(x_sample_resized)

    def representative_dataset():
        for i in range(min(100, len(x_sample_resized))):
            image = x_sample_resized[i]
            image = tf.expand_dims(image, axis=0)  # [1,H,W,C]
            yield [image]

    # ----- 1. Versión FP32 -----
    try:
        converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(modelo)
        tflite_fp32 = converter_fp32.convert()
        with open(f"{nombre_base}_fp32.tflite", "wb") as f:
            f.write(tflite_fp32)
        print(f"✅ Exportado: {nombre_base}_fp32.tflite")
    except Exception as e:
        print("❌ Error exportando FP32:", e)

    # ----- 2. Versión INT8 (cuantización completa) -----
    try:
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(modelo)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = representative_dataset
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.int8
        converter_int8.inference_output_type = tf.int8
        tflite_int8 = converter_int8.convert()
        with open(f"{nombre_base}_int8.tflite", "wb") as f:
            f.write(tflite_int8)
        print(f"✅ Exportado: {nombre_base}_int8.tflite")
    except Exception as e:
        print("⚠️ Error INT8:", e)

    # ----- 3. Versión CH (cuantización híbrida) -----
    try:
        converter_ch = tf.lite.TFLiteConverter.from_keras_model(modelo)
        converter_ch.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_ch = converter_ch.convert()
        with open(f"{nombre_base}_ch.tflite", "wb") as f:
            f.write(tflite_ch)
        print(f"✅ Exportado: {nombre_base}_ch.tflite")
    except Exception as e:
        print("⚠️ Error CH:", e)
