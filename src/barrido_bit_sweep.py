import numpy as np, pandas as pd, random, tensorflow as tf, flatbuffers
from tensorflow.lite.python import schema_py_generated as schema
from evaluate_model import evaluate_model

def barrido_bit_sweep(
    model_content: bytes,
    tensor_name: str,
    x_input: np.ndarray,
    y_true: np.ndarray,
    acc_baseline: float,
    tipo_modelo: str,          # "fp32" | "ch" | "int8"
    seed: int = 123,
    pick_strategy: str = "middle"  # "middle" | "random"
) -> pd.DataFrame:
    """
    Barrido de bits sobre UN tensor: mide degradación por posición de bit.
    - INT8: bits 0..7 de un byte.
    - FP32/CH: bits 0..22 (mantisa) de un float32 (4 bytes).
    Devuelve un DataFrame con una fila por bit probado.
    """
    random.seed(seed); np.random.seed(seed)

    # --- 1) Copia del modelo y localizar el buffer del tensor ---
    mb = bytearray(model_content)
    m = schema.Model.GetRootAsModel(mb, 0)

    target_buffer, tensor_shape = None, None
    for s in range(m.SubgraphsLength()):
        sg = m.Subgraphs(s)
        for t in range(sg.TensorsLength()):
            ten = sg.Tensors(t)
            if ten.Name().decode("utf-8") == tensor_name:
                target_buffer = m.Buffers(ten.Buffer())
                tensor_shape  = tuple(ten.ShapeAsNumpy())
                break
        if target_buffer:
            break
    if target_buffer is None:
        raise ValueError(f"Tensor '{tensor_name}' no encontrado en el FlatBuffer.")

    data = target_buffer.DataAsNumpy()
    if data is None or len(data) == 0:
        raise ValueError(f"Tensor '{tensor_name}' vacío o sin buffer de datos.")

    results = []

    # --- 2) Seleccionar UNA posición base ---
    if tipo_modelo in ("fp32", "ch"):
        if pick_strategy == "middle":
            base = (len(data) // 8) * 4
        else:
            base = (random.randint(0, max(0, len(data) - 4)) // 4) * 4
        if base + 3 >= len(data):
            base = max(0, len(data) - 4)

        bit_positions = list(range(0, 23))   # mantisa
        label_pos = f"float32@byte[{base}:{base+4}]"

    elif tipo_modelo == "int8":
        if pick_strategy == "middle":
            byte_idx = len(data) // 2
        else:
            byte_idx = random.randint(0, len(data) - 1)
        bit_positions = list(range(0, 8))
        label_pos = f"byte@{byte_idx}"
    else:
        raise ValueError("tipo_modelo debe ser 'fp32', 'ch' o 'int8'.")

    # --- 3) Funciones de flip ---
    def _flip_fp32_mantissa(arr, base_offset, mant_bit):
        b = bytes(arr[base_offset:base_offset+4])
        word = int.from_bytes(b, "little", signed=False)
        word ^= (1 << mant_bit)
        arr[base_offset:base_offset+4] = word.to_bytes(4, "little", signed=False)

    def _flip_int8(arr, idx, bit):
        orig = int(arr[idx])
        arr[idx] = orig ^ (1 << bit)
        return orig

    # --- 4) Barrido ---
    for b in bit_positions:
        status = "OK"
        acc_mut = None
        original4, original_byte = None, None

        try:
            if tipo_modelo in ("fp32", "ch"):
                original4 = bytes(data[base:base+4])
                _flip_fp32_mantissa(data, base, b)
            else:
                original_byte = int(data[byte_idx])
                _flip_int8(data, byte_idx, b)

            acc_mut, _ = evaluate_model(
                model_type=tipo_modelo,
                model_content=bytes(mb),
                x_test=x_input,
                y_test=y_true,
                quantized=(tipo_modelo == "int8")
            )

        except Exception as e:
            status = f"Error: {str(e)[:80]}"

        finally:
            if tipo_modelo in ("fp32", "ch") and original4 is not None:
                data[base:base+4] = original4
            elif tipo_modelo == "int8" and original_byte is not None:
                data[byte_idx] = original_byte

        if acc_mut is not None:
            degr = (acc_mut - acc_baseline) * 100.0
            results.append({
                "Tensor": tensor_name,
                "Posición base": label_pos,
                "Bit": b,
                "Acc Baseline (%)": round(float(acc_baseline) * 100, 2),
                "Acc Mutada (%)": round(float(acc_mut) * 100, 2),
                "Degradación (%)": round(float(degr), 2),
                "Estado": status
            })
        else:
            results.append({
                "Tensor": tensor_name,
                "Posición base": label_pos,
                "Bit": b,
                "Acc Baseline (%)": round(float(acc_baseline) * 100, 2),
                "Acc Mutada (%)": None,
                "Degradación (%)": None,
                "Estado": status
            })

    return pd.DataFrame(results).sort_values(by=["Estado", "Bit"])
