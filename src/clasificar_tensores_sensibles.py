import numpy as np
import pandas as pd

def clasificar_tensores_sensibles(df: pd.DataFrame, tipo_modelo: str) -> pd.DataFrame:
    """
    Clasifica tensores como 'SENSIBLE' o 'NO ES SENSIBLE' según el tipo de modelo y su degradación (%).
    
    Reglas:
      - INT8  => sensible si degradación <= -8
      - FP32 / CH => sensible si degradación <= -20
    """
    df = df.copy()

    # Aseguramos que la columna exista y sea numérica
    if "Degradación (%)" not in df.columns:
        raise KeyError("El DataFrame no contiene la columna 'Degradación (%)'")
    
    df["Degradación (%)"] = pd.to_numeric(df["Degradación (%)"], errors="coerce")

    # Normalizamos el tipo de modelo
    tipo_modelo = tipo_modelo.strip().upper()

    if tipo_modelo == "INT8":
        df["TENSOR SENSIBLE?"] = np.where(df["Degradación (%)"] <= -8, "SENSIBLE", "NO ES SENSIBLE")
    elif tipo_modelo in ["FP32", "CH", "FP32/CH"]:
        df["TENSOR SENSIBLE?"] = np.where(df["Degradación (%)"] <= -20, "SENSIBLE", "NO ES SENSIBLE")
    else:
        raise ValueError(f"Tipo de modelo no reconocido: {tipo_modelo}")

    return df
