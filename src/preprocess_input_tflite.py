import tensorflow as tf
import numpy as np

def preprocess_input_tflite(model_name, x_test):
    """
    Preprocesa imágenes redimensionadas para inferencia con modelos TFLite.

    Parámetros:
        model_name (str): 'lenet', 'mobilenetv2', 'googlenet', 'inception', 'efficientnet'
        x_test (np.array): imágenes redimensionadas [n, H, W, 3]

    Retorna:
        x_test_pre (np.array): imágenes listas para inferencia TFLite
    """
    model_name = model_name.lower()

    if model_name == 'lenet':
        x_test_pre = tf.cast(x_test, tf.float32) / 255.0

    elif model_name == 'mobilenetv2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        x_test_pre = tf.cast(x_test, tf.float32)
        x_test_pre = preprocess_input(x_test_pre)  # normaliza a [-1, 1]

    elif model_name in ['googlenet', 'inception']:
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        x_test_pre = tf.cast(x_test, tf.float32)
        x_test_pre = preprocess_input(x_test_pre)

    elif model_name == 'efficientnet':
        from tensorflow.keras.applications.efficientnet import preprocess_input
        x_test_pre = tf.cast(x_test, tf.float32)
        x_test_pre = preprocess_input(x_test_pre)

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    return np.array(x_test_pre)
