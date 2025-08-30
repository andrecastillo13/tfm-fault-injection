import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

def load_imagenet_v2_test(n=1000, target_size=(224, 224)):
    """
    Carga n imágenes del conjunto ImageNet-V2 ya redimensionadas.

    Parámetros:
        n (int): Número de imágenes a cargar (máx 10k)
        target_size (tuple): Tamaño deseado (alto, ancho)

    Retorna:
        x_test (np.array): Imágenes [n, H, W, 3] en float32 sin normalizar
        y_test (np.array): Etiquetas enteras en el rango 0–999
    """
    ds = tfds.load('imagenet_v2', split='test', shuffle_files=False)
    ds = ds.take(n)

    x_test = []
    y_test = []

    for example in tfds.as_numpy(ds):
        image = example["image"]
        label = example["label"]
        # Redimensionar y asegurar tipo float32
        image_resized = tf.image.resize(image, target_size).numpy().astype(np.float32)
        x_test.append(image_resized)
        y_test.append(label)

    x_test = np.stack(x_test)
    y_test = np.array(y_test, dtype=np.int32)

    print(f"✅ Dataset cargado: {x_test.shape[0]} muestras de {target_size[0]}x{target_size[1]}")
    return x_test, y_test
