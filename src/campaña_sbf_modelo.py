import tensorflow as tf
import numpy as np
import pandas as pd
import random
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema
from evaluate_model import evaluate_model  # usamos tu propia evaluación para coherencia

def campaña_sbf_modelo(
    model_content: bytes,
    df_tensores: pd.DataFrame,
    x_input: np.ndarray,
    y_true: np.ndarray,
    acc_baseline: float,
    tipo_modelo: str = "int8",
    N: int = 1
) -> pd.DataFrame:
    """
    Ejecuta una campaña SBF (bit-flip aleatorio) por tensor sobre un modelo TFLite cargado en memoria.
    Compatible con FP32, INT8 y CH, maneja correctamente modelos con batch fijo (EfficientNet).
    
    Parámetros:
    - model_content: contenido binario del modelo .tflite
    - df_tensores: DataFrame con información de tensores inyectables
    - x_input: entrada preprocesada (float32)
    - y_true: etiquetas verdaderas
    - acc_baseline: precisión baseline (0-1) calculada con evaluate_model
    - tipo_modelo: "fp32", "int8" o "ch"
    - N: número de bit-flips aleatorios por tensor

    Retorna:
    - DataFrame con resultados por tensor
    """
    resultados = []

    # Asegurar DataFrame válido
    if not isinstance(df_tensores, pd.DataFrame):
        df_tensores = pd.DataFrame(df_tensores)

    # Iterar por cada tensor inyectable
    for _, row in df_tensores.iterrows():
        tensor_name = row["Nombre"]
        tensor_idx = int(row["IndexModelo"])
        print(f"\n🧬 Inyectando en: {tensor_name} (index={tensor_idx})")

        # Copia del modelo original
        model_bytes = bytearray(model_content)
        model_obj = schema.Model.GetRootAsModel(model_bytes, 0)

        # Buscar buffer del tensor por nombre
        target_buffer = None
        tensor_shape = None
        for s_idx in range(model_obj.SubgraphsLength()):
            subgraph = model_obj.Subgraphs(s_idx)
            for t_idx in range(subgraph.TensorsLength()):
                tensor = subgraph.Tensors(t_idx)
                if tensor.Name().decode("utf-8") == tensor_name:
                    buffer_idx = tensor.Buffer()
                    target_buffer = model_obj.Buffers(buffer_idx)
                    tensor_shape = tuple(tensor.ShapeAsNumpy())
                    break
            if target_buffer:
                break

        if not target_buffer:
            print(f"❌ Tensor '{tensor_name}' no encontrado.")
            continue

        data = target_buffer.DataAsNumpy()
        if data is None or len(data) == 0:
            print(f"⚠️ Tensor vacío: {tensor_name}")
            continue

        # Inyección N bit-flips aleatorios
        pesos_modificados = set()
        for _ in range(max(1, int(N))):
            byte_idx = random.randint(0, len(data) - 1)
            bit_idx = random.randint(0, 7)
            data[byte_idx] ^= (1 << bit_idx)
            pesos_modificados.add(byte_idx)

        try:
            # Evaluar el modelo mutado usando tu evaluate_model
            acc_mutada, _ = evaluate_model(
                model_type=tipo_modelo,
                model_content=bytes(model_bytes),
                x_test=x_input,
                y_test=y_true,
                quantized=(tipo_modelo == "int8")
            )

            delta = acc_mutada - acc_baseline
            resultados.append({
                "Index Modelo": tensor_idx,
                "Tensor": tensor_name,
                "Forma": tensor_shape,
                "# Pesos": int(row.get("# Pesos", np.prod(tensor_shape) if tensor_shape else 0)),
                "Total Bits Inyectables": int(row.get("Total Bits Inyectables", len(data)*8)),
                "Pesos Modificados": int(len(pesos_modificados)),
                "Acc Baseline": round(float(acc_baseline) * 100, 2),
                "Acc Post-inyección": round(float(acc_mutada) * 100, 2),
                "Degradación (%)": round(delta * 100, 2)
            })

        except Exception as e:
            print(f"❌ Error durante inferencia en tensor '{tensor_name}': {e}")
            continue

    if not resultados:
        return pd.DataFrame(columns=[
            "Index Modelo","Tensor","Forma","# Pesos","Total Bits Inyectables",
            "Pesos Modificados","Acc Baseline","Acc Post-inyección","Degradación (%)"
        ])

    return pd.DataFrame(resultados).sort_values(by="Degradación (%)")
