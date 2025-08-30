
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema
from evaluate_model import evaluate_model  # coherente con tu pipeline

def bit_sweep_modelo(
    model_content: bytes,
    tensor_name: str,
    x_input: np.ndarray,
    y_true: np.ndarray,
    acc_baseline: float,
    tipo_modelo: str = "int8",
    k_flips_por_bit: int = 1,
    bits_por_peso: int | None = None,
) -> pd.DataFrame:
    """
    Barrido de bits (Bit Sweep) sobre *un único tensor* de un modelo TFLite cargado en memoria.

    Idea (alineada con SBF):
      - Se ejecuta 1 experimento por posición de bit.
      - En cada experimento se hace un flip del bit en 'k_flips_por_bit' pesos aleatorios del tensor objetivo.
      - Se evalúa el modelo y se mide la degradación respecto al baseline.

    Parámetros:
      - model_content: bytes del .tflite original (sin modificar en disco).
      - tensor_name: nombre EXACTO del tensor a barrer (por ejemplo, "arith.constant9" o "sequential/conv2d/Conv2D;.../Const").
      - x_input, y_true: dataset ya preprocesado para el modelo (coherente con evaluate_model).
      - acc_baseline: precisión baseline (0..1) del mismo modelo y dataset.
      - tipo_modelo: "int8", "fp32" o "ch" (se usa para llamar a evaluate_model y para asumir tamaño de peso).
      - k_flips_por_bit: cuántos pesos aleatorios del tensor se verán afectados por cada posición de bit.
      - bits_por_peso: override manual. Si None → 8 para "int8", 32 para "fp32/ch".

    Retorna:
      - DataFrame con una fila por posición de bit barrida. Columnas principales:
          ["Tensor","Bit Pos","Flips por Bit","Acc Baseline","Acc Post-Bit","Degradación (%)","Notas"]
    """
    # ---- Determinar bits por peso según tipo de modelo ----
    tipo = (tipo_modelo or "").strip().lower()
    if bits_por_peso is None:
        bits_por_peso = 8 if tipo == "int8" else 32

    if k_flips_por_bit < 1:
        k_flips_por_bit = 1

    # ---- Parsear FlatBuffer y localizar el tensor objetivo ----
    model_bytes_master = bytearray(model_content)  # copia maestra inalterada
    model_obj = schema.Model.GetRootAsModel(model_bytes_master, 0)

    target_buffer = None
    tensor_shape = None
    tensor_size = None  # nº de pesos (elementos)
    peso_bytes = 1 if bits_por_peso == 8 else 4  # asunción coherente con int8 vs fp32/ch

    for s_idx in range(model_obj.SubgraphsLength()):
        subgraph = model_obj.Subgraphs(s_idx)
        for t_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(t_idx)
            name = tensor.Name().decode("utf-8")
            if name == tensor_name:
                buffer_idx = tensor.Buffer()
                target_buffer = model_obj.Buffers(buffer_idx)
                tensor_shape = tuple(tensor.ShapeAsNumpy())
                # nº de elementos (pesos) = producto de la forma
                try:
                    tensor_size = int(np.prod(tensor_shape)) if len(tensor_shape) > 0 else 0
                except Exception:
                    tensor_size = None
                break
        if target_buffer:
            break

    if target_buffer is None:
        raise ValueError(f"Tensor '{tensor_name}' no encontrado dentro del modelo.")

    data_master = target_buffer.DataAsNumpy()
    if data_master is None or len(data_master) == 0:
        raise ValueError(f"Tensor '{tensor_name}' vacío o sin datos inyectables.")

    if tensor_size is None or tensor_size <= 0:
        # Fallback: estimar a partir del tamaño del buffer y bytes por peso
        tensor_size = len(data_master) // peso_bytes

    # ---- Definir el rango de bits a barrer ----
    rango_bits = range(bits_por_peso)  # int8: 0..7, fp32/ch: 0..31

    resultados = []

    # ---- Función auxiliar para elegir 'k' índices de peso aleatorios distintos ----
    def elegir_indices_peso(k, max_elem):
        k = min(k, max_elem)
        return random.sample(range(max_elem), k) if k > 0 else []

    for bit_pos in rango_bits:
        # Copiar bytes del modelo base (para que cada bit se evalúe de forma independiente)
        model_bytes = bytearray(model_bytes_master)  # copia del modelo completo
        model_obj_local = schema.Model.GetRootAsModel(model_bytes, 0)

        # Acceder al mismo buffer del tensor en esta copia
        # Nota: FlatBuffers re-map; volvemos a tomar el puntero al buffer en la copia
        target_buffer_local = None
        for s_idx in range(model_obj_local.SubgraphsLength()):
            subgraph = model_obj_local.Subgraphs(s_idx)
            for t_idx in range(subgraph.TensorsLength()):
                tensor = subgraph.Tensors(t_idx)
                name = tensor.Name().decode("utf-8")
                if name == tensor_name:
                    target_buffer_local = model_obj_local.Buffers(tensor.Buffer())
                    break
            if target_buffer_local:
                break

        if target_buffer_local is None:
            resultados.append({
                "Tensor": tensor_name,
                "Bit Pos": bit_pos,
                "Flips por Bit": 0,
                "Acc Baseline": round(float(acc_baseline) * 100, 4),
                "Acc Post-Bit": None,
                "Degradación (%)": None,
                "Notas": "No se pudo mapear el buffer en la copia."
            })
            continue

        data = target_buffer_local.DataAsNumpy()
        if data is None or len(data) == 0:
            resultados.append({
                "Tensor": tensor_name,
                "Bit Pos": bit_pos,
                "Flips por Bit": 0,
                "Acc Baseline": round(float(acc_baseline) * 100, 4),
                "Acc Post-Bit": None,
                "Degradación (%)": None,
                "Notas": "Buffer vacío en la copia."
            })
            continue

        # ---- Seleccionar 'k' pesos aleatorios y mutar el bit 'bit_pos' en cada uno ----
        pesos_a_mutar = elegir_indices_peso(k_flips_por_bit, tensor_size)
        notas = []

        for w_idx in pesos_a_mutar:
            # byte base del peso dentro del buffer
            base_byte = w_idx * peso_bytes

            if bits_por_peso == 8:
                # Bit dentro del único byte
                byte_idx = base_byte
                mask = (1 << bit_pos)
                if byte_idx < len(data):
                    data[byte_idx] ^= mask
                else:
                    notas.append(f"byte_idx fuera de rango ({byte_idx})")
            else:
                # fp32/ch: mapa bit_pos (0..31) → (byte dentro del peso, bit dentro del byte)
                byte_in_weight = bit_pos // 8      # 0..3
                bit_in_byte   = bit_pos % 8        # 0..7
                byte_idx      = base_byte + byte_in_weight
                mask = (1 << bit_in_byte)
                if byte_idx < len(data):
                    data[byte_idx] ^= mask
                else:
                    notas.append(f"byte_idx fuera de rango ({byte_idx})")

        # ---- Evaluar el modelo mutado ----
        try:
            acc_mutada, _ = evaluate_model(
                model_type=tipo,
                model_content=bytes(model_bytes),
                x_test=x_input,
                y_test=y_true,
                quantized=(tipo == "int8")
            )
            delta = acc_mutada - acc_baseline
            resultados.append({
                "Tensor": tensor_name,
                "Bit Pos": int(bit_pos),
                "Flips por Bit": int(len(pesos_a_mutar)),
                "Acc Baseline": round(float(acc_baseline) * 100, 4),
                "Acc Post-Bit": round(float(acc_mutada) * 100, 4),
                "Degradación (%)": round(delta * 100, 4),
                "Notas": "; ".join(notas) if notas else ""
            })
        except Exception as e:
            resultados.append({
                "Tensor": tensor_name,
                "Bit Pos": int(bit_pos),
                "Flips por Bit": int(len(pesos_a_mutar)),
                "Acc Baseline": round(float(acc_baseline) * 100, 4),
                "Acc Post-Bit": None,
                "Degradación (%)": None,
                "Notas": f"Error en inferencia: {e}"
            })

    df = pd.DataFrame(resultados)

    # ---- Ordenar por posición de bit (asc) ----
    if "Bit Pos" in df.columns:
        df = df.sort_values(by="Bit Pos").reset_index(drop=True)

    # ---- Resumen por consola (estilo SBF) ----
    try:
        # Acc post-barrido: podemos reportar la peor (mínima) o la media. Optamos por la media para un resumen estable.
        acc_media = pd.to_numeric(df["Acc Post-Bit"], errors="coerce").dropna().mean()
        if pd.notna(acc_media):
            degrad_media = (acc_media/100.0 - acc_baseline) * 100.0
            print(f"🔎 Bit Sweep → Tensor: {tensor_name}")
            print(f"   • Acc Baseline: {acc_baseline*100:.4f}%")
            print(f"   • Acc Post-Barrido (media): {acc_media:.4f}%")
            print(f"   • Degradación media: {degrad_media:.4f}%")
    except Exception:
        pass

    return df
