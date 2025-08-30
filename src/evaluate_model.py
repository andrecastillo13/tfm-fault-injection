import tensorflow as tf
import numpy as np

def evaluate_model(model_type, model_content, x_test, y_test, quantized=False):
    """
    Evalúa un modelo TFLite cargado en memoria (no desde archivo .tflite).

    Parámetros:
    - model_type (str): "fp32", "int8" o "ch"
    - model_content (bytes): contenido binario del modelo TFLite
    - x_test (np.array): imágenes ya preprocesadas y redimensionadas
    - y_test (np.array): etiquetas enteras (int)
    - quantized (bool): True si el modelo requiere cuantización de entrada

    Retorna:
    - acc (float): precisión top-1
    - y_pred (list): predicciones como enteros
    """
    y_pred = []

    # 🔁 Aceptar tanto rutas como contenido binario
    if isinstance(model_content, str):
        interpreter = tf.lite.Interpreter(model_path=model_content)
    elif isinstance(model_content, (bytes, bytearray)):
        interpreter = tf.lite.Interpreter(model_content=model_content)
    else:
        raise ValueError("El argumento 'model_content' debe ser una ruta (str) o bytes.")

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if quantized:
        input_scale, input_zero_point = input_details[0]['quantization']

    for sample in x_test:
        if quantized:
            sample_q = (sample / input_scale + input_zero_point).astype(np.int8)
        else:
            sample_q = sample.astype(np.float32)

        if sample_q.ndim == 3:
            sample_q = np.expand_dims(sample_q, axis=0)

        interpreter.set_tensor(input_details[0]['index'], sample_q)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(output))

    y_pred = np.array(y_pred)
    acc = np.mean(y_pred == y_test[:len(y_pred)])
    print(f"✅ Modelo: {model_type.upper()}  → 🎯 Accuracy: {acc:.4f}")

    return acc, y_pred