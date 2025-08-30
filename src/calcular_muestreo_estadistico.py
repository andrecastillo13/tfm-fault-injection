import math

def calcular_muestreo_estadistico(N=None, Z=1.96, p=0.5, e=0.05):
    """
    Calcula el tamaño de muestra necesario para una población INFINITA
    bajo un nivel de confianza y error específico.

    Parámetros:
    - Z: valor crítico para el nivel de confianza (default 1.96 para 95%)
    - p: proporción esperada de éxito (default 0.5)
    - e: margen de error permitido (default 0.05)

    Retorna:
    - int: tamaño de la muestra (n)
    """
    n = (Z**2) * p * (1 - p) / (e**2)
    return math.ceil(n)
