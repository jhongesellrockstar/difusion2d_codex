# difusion2d_codex

Este proyecto acompaña al miniartículo donde se comparan tres esquemas explícitos para resolver la ecuación de difusión en dos dimensiones. El objetivo del paper es analizar la precisión y el costo computacional de cada método al ejecutarse en hardware modesto.

Los esquemas implementados son:

* **FTCS (5 puntos)**: la aproximación en cruz clásica.
* **9 puntos**: extiende la plantilla con los nodos diagonales.
* **(1,13) de Dehghan**: un esquema de 13 puntos que utiliza extrapolación para las fronteras.

## Requisitos

Se necesita Python 3.11 o superior. Las dependencias básicas pueden instalarse con:

```bash
pip install -r requirements.txt
```

Si se desea recrear el entorno completo de desarrollo puede usarse el archivo `environment_NSI_miniarticulo_01.txt` con Conda:

```bash
conda create -n difusion2d --file environment_NSI_miniarticulo_01.txt
```

## Ejecución

El script `scripts/main_benchmark.py` ejecuta los tres solvers en una malla moderada y compara su tiempo y precisión. Para correrlo:

```bash
python scripts/main_benchmark.py
```

Puede agregarse `--visualize` para mostrar un contorno de la solución obtenida con el método de 13 puntos.

## Ejemplo de salida

Al finalizar se imprime una tabla similar a la siguiente:

```
Benchmark results:
     Scheme     Time (s)     L2 error
       FTCS      0.0935   1.4e-03
    9-point      0.0801   4.8e-04
   13-point      0.1078   2.3e-04
```

Además se genera una gráfica comparativa de tiempo y error.

## Licencia

El código se distribuye bajo la licencia GPLv3, consulte el archivo `LICENSE` para más detalles.
