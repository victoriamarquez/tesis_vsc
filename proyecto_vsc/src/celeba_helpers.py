import logging
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image


def load_celeba_attributes(attr_path):
    """Carga los atributos faciales del dataset CelebA desde el archivo de texto provisto.

    El archivo de atributos se espera que tenga el formato estándar de CelebA,
    donde las primeras dos líneas contienen metadatos y la tercera línea comienza
    con los datos de las imágenes (nombre de archivo seguido de los atributos binarios).

    Args:
        attr_path (str): Ruta al archivo de texto que contiene los atributos de CelebA.

    Returns:
        pandas.DataFrame: Un DataFrame con la columna 'file_name' y una columna
            para cada uno de los atributos faciales (ej. 'Male', 'Smiling'),
            con valores -1 (Ausente) o 1 (Presente).
    """
    logging.info("[CelebA] [→] Cargando atributos CelebA desde archivo.")
    with open(attr_path, "r") as attribute_file:
        lines = attribute_file.readlines()
    attribute_names = lines[1].strip().split()
    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        filename = parts[0]
        attributes = list(map(int, parts[1:]))
        data.append([filename] + attributes)
    dataframe = pd.DataFrame(data, columns=["file_name"] + attribute_names)
    logging.info("[CelebA] [✔] Atributos CelebA cargados desde archivo.")
    return dataframe


def align_celeba_candidates(input_dir, output_dir):
    """Alinea imágenes de CelebA usando el alineador de StyleGAN (estilo FFHQ).

    Necesario para que las imágenes de CelebA tengan el mismo preprocesamiento
    que las de BU-3DFE antes de ser proyectadas al espacio latente.

    Args:
        input_dir (str): Carpeta con las imágenes de CelebA a alinear.
        output_dir (str): Carpeta donde guardar las imágenes alineadas (1024×1024).
    """
    script_path = "/home/vicky/Documents/tesis_vsc/stylegan2encoder/align_images.py"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info(f"[CelebA] [→] Alineando candidatas desde {input_dir}.")
    command = ["python3", script_path, str(input_dir), str(output_dir)]
    subprocess.run(command, check=True)
    logging.info(f"[CelebA] [✔] Alineación completada. Imágenes en {output_dir}.")


def replace_background_with_black(input_dir, output_dir):
    """Reemplaza el fondo de las imágenes alineadas con negro sólido usando rembg.

    Útil para aislar el efecto del fondo en la proyección y modificación emocional:
    las imágenes de BU-3DFE tienen fondo negro, y queremos ver si homogeneizar el
    fondo de CelebA mejora los resultados.

    rembg se importa de forma lazy para no romper el entorno si no está instalado.

    Args:
        input_dir (str): Carpeta con las imágenes alineadas (.png).
        output_dir (str): Carpeta donde guardar las imágenes con fondo negro.
    """
    from rembg import remove

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_paths = sorted(Path(input_dir).glob("*.png"))
    total = len(image_paths)

    logging.info(f"[CelebA] [→] Reemplazando fondo con negro para {total} imágenes.")
    for index, image_path in enumerate(image_paths, start=1):
        original = Image.open(image_path)
        without_background = remove(original)

        black_background = Image.new("RGB", without_background.size, (0, 0, 0))
        black_background.paste(without_background, mask=without_background.split()[3])

        black_background.save(Path(output_dir) / image_path.name)
        logging.info(f"[CelebA] [✔] [{index}/{total}] Fondo reemplazado: {image_path.name}.")

    logging.info(f"[CelebA] [✔] Reemplazo de fondo completado. Imágenes en {output_dir}.")
