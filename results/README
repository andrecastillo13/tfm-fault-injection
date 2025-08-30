# Carpeta `results/`

En esta carpeta se almacenan los **resultados experimentales** generados por las campañas de inyección de fallos.  
Los resultados incluyen métricas en formato CSV y visualizaciones asociadas (gráficos, matrices de confusión, etc.).

---

## Estructura recomendada

```
results/
├── lenet5/
│   ├── sbf/        # Resultados de Single Bit Flip
│   ├── bitsweep/   # Resultados de barridos por posición de bit
│   └── mbf/        # Resultados de Multiple Bit Flips
├── mobilenetv2/
│   ├── sbf/
│   ├── bitsweep/
│   └── mbf/
├── inceptionv3/
│   ├── sbf/
│   ├── bitsweep/
│   └── mbf/
├── efficientnetb0/
│   ├── sbf/
│   ├── bitsweep/
│   └── mbf/
└── summary/        # Tablas comparativas y gráficos agregados
```

---

## Formato de los ficheros CSV

Cada CSV generado por las campañas contiene, como mínimo, las siguientes columnas:

- **model** → nombre del modelo (`lenet5`, `mobilenetv2`, `inceptionv3`, `efficientnetb0`)  
- **variant** → representación (`fp32`, `ch`, `int8`)  
- **tensor** → nombre del tensor inyectado  
- **exp_type** → tipo de experimento (`sbf`, `bitsweep`, `mbf`)  
- **N / k / K** → número de flips o parámetro correspondiente al experimento  
- **seed** → semilla utilizada para reproducibilidad  
- **acc_base** → precisión baseline del modelo antes de la inyección  
- **acc_post** → precisión tras la inyección  
- **delta_acc** → degradación en precisión respecto a la baseline  

Ejemplo de cabecera típica:

```csv
model,variant,tensor,exp_type,N,K,seed,acc_base,acc_post,delta_acc
lenet5,int8,conv2d_1/weights,sbf,1,,42,0.6334,0.5921,-0.0413
```

---

## Buenas prácticas

- Mantener esta carpeta fuera de control de versiones (`.gitignore`) excepto por:
  - `README.md`
  - `samples/` (ejemplos ligeros de resultados)  
- Versionar solo tablas pequeñas de resumen o ejemplos ilustrativos.  
- Los resultados completos se regeneran al ejecutar los cuadernos o los scripts de `src/`.

---
