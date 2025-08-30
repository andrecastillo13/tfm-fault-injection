import tensorflow as tf
import numpy as np
import pandas as pd
from calcular_muestreo_estadistico import calcular_muestreo_estadistico

def inspeccionar_tensores_inyectables(model_content, tipo="fp32"):
    """
    Inspecciona los tensores inyectables de un modelo TFLite cargado en memoria o desde ruta.

    Parámetros:
    - model_content (bytes o str): contenido binario del modelo TFLite o ruta
    - tipo (str): "fp32", "ch" o "int8"

    Retorna:
    - DataFrame con información de los tensores inyectables
    """
    # 🛠 Aceptar tanto bytes como ruta
    if isinstance(model_content, str):
        interpreter = tf.lite.Interpreter(model_path=model_content)
    elif isinstance(model_content, (bytes, bytearray)):
        interpreter = tf.lite.Interpreter(model_content=model_content)
    else:
        raise ValueError("El argumento 'model_content' debe ser una ruta (str) o bytes.")

    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()

    tensores_info = []

    for detail in tensor_details:
        idx = detail["index"]
        dtype = detail["dtype"]
        name = detail["name"]
        shape = detail["shape"]

        try:
            tensor = interpreter.get_tensor(idx)
        except Exception:
            continue

        if tensor.size == 0:
            continue

        es_fp32 = (tipo == "fp32" and dtype == np.float32)
        es_int8 = (tipo == "int8" and dtype == np.int8)
        es_ch   = (tipo == "ch" and dtype in [np.float32, np.int8])

        if es_fp32 or es_int8 or es_ch:
            if tipo == "int8":
                es_inyectable = any(kw in name for kw in ["pseudo_qconst", "const", "kernel", "weights"])
            else:
                es_inyectable = any(kw in name for kw in ["const", "kernel", "weights"])

            if es_inyectable:
                bits_por_peso = 32 if dtype == np.float32 else 8
                total_bits = tensor.size * bits_por_peso
                n = calcular_muestreo_estadistico(total_bits)

                tensores_info.append({
                    "IndexModelo": idx,
                    "Nombre": name,
                    "TipoDato": str(dtype),
                    "Forma": shape,
                    "# Pesos": tensor.size,
                    "Bits por Peso": bits_por_peso,
                    "Total Bits Inyectables": total_bits,
                    "n (muestra estadística)": n
                })

    df = pd.DataFrame(tensores_info)
    df = df.sort_values(by="IndexModelo").reset_index(drop=True)
    return df
