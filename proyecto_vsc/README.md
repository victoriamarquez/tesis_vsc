# Tesis de Licenciatura — Generación de Expresiones Faciales con StyleGAN2

Herramienta que, a partir de una foto neutra de una persona, genera versiones
sintéticas de su rostro con distintas emociones (felicidad, enojo, disgusto,
miedo, tristeza, sorpresa) en intensidad controlada.

El objetivo clínico es facilitar pruebas de reconocimiento emocional en
pacientes bajo intervención neuroquirúrgica.

## ¿Cómo funciona?

```
Dataset BU-3DFE (rostros con emociones graduadas)
        │
        ▼
Proyección al espacio latente W+ de StyleGAN2
        │
        ▼
Cálculo de direcciones emocionales (PCA / Regresión Lineal)
        │
        ▼
w_modificado = w_neutro + intensidad × dirección_emoción
        │
        ▼
StyleGAN2 genera la imagen resultante
```

## Requisitos

- Docker (para ejecutar StyleGAN2-ADA-PyTorch)
- Python 3.x con dependencias del entorno `entorno_stylegan.yml`
- Dataset BU-3DFE (imágenes `.bmp` organizadas por persona)
- Dataset CelebA (para pruebas de generalización)

```bash
conda env create -f entorno_stylegan.yml
conda activate <nombre_entorno>
cd proyecto_vsc/src
```

## Uso

El punto de entrada es `argument_parsing.py`, con tres modos:

### 1. Calcular vectores emocionales

Procesa el dataset BU-3DFE, proyecta las imágenes al espacio latente y calcula
las direcciones emocionales. El resultado se guarda en `datos/directions_regression.csv`.

```bash
python argument_parsing.py calculate_vectors
```

> Las flags `align`, `process` y `generate` dentro de `calculate_vectors.py`
> controlan si se re-ejecutan la alineación, proyección y generación de imágenes
> (por defecto solo se proyecta).

### 2. Modificar una imagen nueva

Toma una carpeta con una o más imágenes, las alinea, las proyecta y genera
versiones con las emociones aplicadas.

```bash
# Todas las emociones con intensidades por defecto
python argument_parsing.py modify_image /ruta/a/carpeta/

# Una emoción específica con intensidad personalizada
python argument_parsing.py modify_image /ruta/a/carpeta/ --emotion HA --intensity 3.5
```

Emociones disponibles: `HA` (felicidad), `AN` (enojo), `DI` (disgusto),
`FE` (miedo), `SA` (tristeza), `SU` (sorpresa).

### 3. Ejecutar pruebas

Aplica los vectores emocionales sobre un subconjunto diverso del dataset propio
y sobre imágenes neutras de CelebA.

```bash
python argument_parsing.py test
```

### Opciones globales

| Flag | Descripción |
|------|-------------|
| `-v` / `--verbose` | Muestra logs en consola |
| `--no-logging` | Desactiva el registro de eventos |

## Estructura del proyecto

```
proyecto_vsc/src/
├── argument_parsing.py      # Punto de entrada (CLI)
├── calculate_vectors.py     # Pipeline de cálculo de direcciones
├── modify_image.py          # Modificación de imágenes nuevas
├── testing.py               # Pruebas sobre datasets propios y CelebA
├── image_processing.py      # Alineación, proyección y generación (compartido)
├── pca.py                   # Método PCA para calcular direcciones
├── lineal_regression.py     # Método de regresión lineal para direcciones
├── method_comparison.py     # Comparación coseno entre métodos
├── diverse_group_testing.py # Pruebas con subconjunto balanceado
├── celeba_processing.py     # Procesamiento del dataset CelebA
└── helpers.py               # Funciones auxiliares compartidas
datos/
├── directions_regression.csv  # Vectores de dirección (output de Parte 1)
└── metadatos_con_vectores.pkl # Metadatos del dataset con vectores latentes
```
