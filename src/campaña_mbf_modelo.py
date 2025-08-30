
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import flatbuffers
from typing import List, Optional
from tensorflow.lite.python import schema_py_generated as schema
from evaluate_model import evaluate_model  # coherente con tu pipeline

def _get_tensor_name_and_index(row: pd.Series):
    """Admite DataFrames provenientes de 'inspeccionar_tensores_inyectables' o de 'clasificar_tensores_sensibles'."""
    name = row.get("Nombre", None)
    if name is None:
        name = row.get("Tensor", None)
    idx = row.get("IndexModelo", None)
    if idx is None:
        idx = row.get("Index Modelo", None)
    return name, None if idx is None else int(idx)

def _find_buffer_by_tensor_name(model_bytes: bytearray, tensor_name: str):
    """Localiza y retorna (buffer, shape_tuple) del tensor por nombre dentro del FlatBuffer."""
    model_obj = schema.Model.GetRootAsModel(model_bytes, 0)
    for s_idx in range(model_obj.SubgraphsLength()):
        subgraph = model_obj.Subgraphs(s_idx)
        for t_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(t_idx)
            name = tensor.Name().decode("utf-8")
            if name == tensor_name:
                buffer_idx = tensor.Buffer()
                return model_obj.Buffers(buffer_idx), tuple(tensor.ShapeAsNumpy())
    return None, None

def _flip_many_bits_inplace(data: np.ndarray, num_flips: int):
    """
    Realiza 'num_flips' flips aleatorios (byte_idx, bit_idx) dentro del buffer plano 'data' IN-PLACE.
    Evita repetir el mismo (byte_idx, bit_idx) para no anular el flip.
    """
    total_bytes = len(data)
    if total_bytes == 0 or num_flips <= 0:
        return 0

    max_unique_pairs = total_bytes * 8
    flips_objetivo = int(min(num_flips, max_unique_pairs))
    visitados = set()
    hechos = 0

    while hechos < flips_objetivo:
        byte_idx = random.randint(0, total_bytes - 1)
        bit_idx = random.randint(0, 7)
        par = (byte_idx, bit_idx)
        if par in visitados:
            continue
        visitados.add(par)
        data[byte_idx] ^= (1 << bit_idx)
        hechos += 1

    return hechos

def campaña_mbf_modelo(
    model_content: bytes,
    df_tensores: pd.DataFrame,
    x_input: np.ndarray,
    y_true: np.ndarray,
    acc_baseline: float,
    tipo_modelo: str = "int8",
    progresion: Optional[List[int]] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Campaña MBF (Multiple Bit Flips) sobre *tensores sensibles*.
    Para cada tensor se evaluará una progresión de #flips simultáneos: por defecto [1, 25, 50, 75, 100].
    """
    if random_seed is not None:
        random.seed(int(random_seed))
        np.random.seed(int(random_seed))

    if progresion is None:
        progresion = [1, 25, 50, 75, 100]

    if not isinstance(df_tensores, pd.DataFrame):
        df_tensores = pd.DataFrame(df_tensores)

    resultados = []

    for _, row in df_tensores.iterrows():
        tensor_name, tensor_idx = _get_tensor_name_and_index(row)
        if tensor_name is None:
            continue

        print(f"\n🧪 MBF en tensor: {tensor_name}")

        model_bytes_master = bytearray(model_content)

        target_buffer_master, tensor_shape = _find_buffer_by_tensor_name(model_bytes_master, tensor_name)
        if target_buffer_master is None:
            print(f"  ⚠️ Buffer no encontrado: {tensor_name}")
            continue

        data_master = target_buffer_master.DataAsNumpy()
        if data_master is None or len(data_master) == 0:
            print(f"  ⚠️ Tensor vacío: {tensor_name}")
            continue

        num_pesos = int(row.get("# Pesos", 0)) if "# Pesos" in row else (len(data_master))
        total_bits_inyectables = int(row.get("Total Bits Inyectables", len(data_master) * 8))

        for K in progresion:
            model_bytes = bytearray(model_bytes_master)

            target_buffer, _ = _find_buffer_by_tensor_name(model_bytes, tensor_name)
            if target_buffer is None:
                resultados.append({
                    "Index Modelo": tensor_idx,
                    "Tensor": tensor_name,
                    "Forma": tensor_shape,
                    "# Pesos": num_pesos,
                    "Total Bits Inyectables": total_bits_inyectables,
                    "K (flips simultáneos)": int(K),
                    "Acc Baseline": round(float(acc_baseline)*100, 4),
                    "Acc Post-inyección": None,
                    "Degradación (%)": None,
                    "Notas": "No fue posible mapear buffer en copia"
                })
                continue

            data = target_buffer.DataAsNumpy()
            if data is None or len(data) == 0:
                resultados.append({
                    "Index Modelo": tensor_idx,
                    "Tensor": tensor_name,
                    "Forma": tensor_shape,
                    "# Pesos": num_pesos,
                    "Total Bits Inyectables": total_bits_inyectables,
                    "K (flips simultáneos)": int(K),
                    "Acc Baseline": round(float(acc_baseline)*100, 4),
                    "Acc Post-inyección": None,
                    "Degradación (%)": None,
                    "Notas": "Buffer vacío en copia"
                })
                continue

            flips_realizados = _flip_many_bits_inplace(data, int(K))

            try:
                acc_mutada, _ = evaluate_model(
                    model_type=tipo_modelo,
                    model_content=bytes(model_bytes),
                    x_test=x_input,
                    y_test=y_true,
                    quantized=(tipo_modelo.lower() == "int8")
                )
                delta = acc_mutada - acc_baseline
                resultados.append({
                    "Index Modelo": tensor_idx,
                    "Tensor": tensor_name,
                    "Forma": tensor_shape,
                    "# Pesos": num_pesos,
                    "Total Bits Inyectables": total_bits_inyectables,
                    "K (flips simultáneos)": int(flips_realizados),
                    "Acc Baseline": round(float(acc_baseline)*100, 4),
                    "Acc Post-inyección": round(float(acc_mutada)*100, 4),
                    "Degradación (%)": round(delta*100, 4),
                    "Notas": ""
                })
            except Exception as e:
                resultados.append({
                    "Index Modelo": tensor_idx,
                    "Tensor": tensor_name,
                    "Forma": tensor_shape,
                    "# Pesos": num_pesos,
                    "Total Bits Inyectables": total_bits_inyectables,
                    "K (flips simultáneos)": int(flips_realizados),
                    "Acc Baseline": round(float(acc_baseline)*100, 4),
                    "Acc Post-inyección": None,
                    "Degradación (%)": None,
                    "Notas": f"Error en inferencia: {e}"
                })

        print(f"  ✅ MBF completado en {tensor_name}.")

    if not resultados:
        return pd.DataFrame(columns=[
            "Index Modelo","Tensor","Forma","# Pesos","Total Bits Inyectables",
            "K (flips simultáneos)","Acc Baseline","Acc Post-inyección","Degradación (%)","Notas"
        ])

    df = pd.DataFrame(resultados)
    if "K (flips simultáneos)" in df.columns:
        df = df.sort_values(by=["Tensor","K (flips simultáneos)"]).reset_index(drop=True)
    return df
