import os
import subprocess
import pandas as pd
import numpy as np
from numpy import load, savez
from pathlib import Path
from glob import glob

from image_processing import generate_one_image_from_npz


def load_celeba_attributes(attr_path):
    print("FUNCT: load_celeba_attributes")
    with open(attr_path, 'r') as f:
        lines = f.readlines()

    # La segunda línea contiene los nombres de los atributos
    attribute_names = lines[1].strip().split()

    # Las siguientes líneas contienen los datos: filename seguido de los atributos
    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        filename = parts[0]
        attributes = list(map(int, parts[1:]))  # convierte de str a int
        data.append([filename] + attributes)

    # Crear el DataFrame
    df = pd.DataFrame(data, columns=['file_name'] + attribute_names)
    return df


def process_selected_celeba_images_from_df(df, steps, verbose=False):
    print("FUNC: process_selected_celeba_images_from_df")
    # Path base en el host
    base_input_host = "/home/vicky/Documents/tesis_vsc/images/CelebA/img_align_celeba"
    base_output_host = "/home/vicky/Documents/tesis_vsc/images/CelebA/img_align_celeba"

    # Path que ve el contenedor
    base_input_docker = "/scratch/CelebA/img_align_celeba"
    base_output_docker = "/scratch/CelebA/npz_aligned_celeba"

    # Crear output dir si no existe
    os.makedirs(base_output_host, exist_ok=True)

    # Comando base
    command_base = [
        "/home/vicky/Documents/tesis_vsc/stylegan2-ada-pytorch/docker_run.sh",
        "python", "stylegan2-ada-pytorch/projector.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        f"--num-steps={steps}",
        "--seed=303",
        "--save-video=False"
    ]

    # Iterar por imágenes en el dataframe
    for file_name in df["file_name"]:
        target_path = os.path.join(base_input_docker, file_name)    # dentro del contenedor
        outdir_path = base_output_docker                            # dentro del contenedor

        command = command_base + [
            f"--target={target_path}",
            f"--outdir={outdir_path}"
        ]

        if verbose:
            print(f"Procesando {file_name}")
            print(f"Comando: {' '.join(command)}")

        subprocess.run(command, check=True)

    if verbose:
        print("✔ Proyección completa para imágenes de CelebA seleccionadas.")

def modify_latent_with_emotion(npz_path, emotion, emotion_vectors, multiplier, output_dir):
    """
    Toma un archivo .npz de CelebA, le suma el vector emocional y guarda un nuevo archivo .npz.

    Args:
        npz_path (str): Ruta al archivo .npz original.
        emotion (str): Código de emoción, ej. 'HA', 'AN'.
        emotion_vectors (dict): Diccionario con vectores emocionales. Ej: {'HA': np.array([...]), ...}
        multiplier (float): Escalar que multiplica el vector emocional.
        output_dir (str): Carpeta donde guardar el nuevo .npz.
    """
    # Cargar vector original (w tiene shape (1, 18, 512))
    w = load(npz_path)['w']
    
    # Sumar el vector emocional (broadcast en las 18 capas)
    delta = emotion_vectors[emotion] * multiplier  # (512,)
    delta_expanded = np.tile(delta.reshape(1, 1, 512), (1, 18, 1))  # (1, 18, 512)
    w_modificado = w + delta_expanded

    # Crear nombre nuevo para el archivo
    base_name = Path(npz_path).stem  # ej. "000003_01_projected_w"
    nombre_salida = f"{base_name}_emo_{emotion}.npz"
    path_salida = os.path.join(output_dir, nombre_salida)

    # Guardar nuevo archivo
    os.makedirs(output_dir, exist_ok=True)
    savez(path_salida, w=w_modificado)
    return path_salida  # lo devolvemos para usarlo después con generate_one_image_from_npz


def process_emotions_celeba(npz_input_dir, emotion_vectors, multiplicadores, output_npz_dir, max_imagenes=None):
    """
    Modifica vectores latentes con emociones y genera imágenes correspondientes.

    Args:
        npz_input_dir (str): Carpeta con los archivos .npz originales de CelebA.
        emotion_vectors (dict): Diccionario con vectores emocionales (por ejemplo, obtenidos por regresión).
        multiplicadores (dict): Diccionario con multiplicadores por emoción. Ej: {'HA': 3, 'AN': 4, ...}
        output_npz_dir (str): Carpeta donde guardar los nuevos .npz modificados.
        max_imagenes (int): Si se desea limitar la cantidad de imágenes procesadas (por ejemplo, 10).
    """
    # Obtener lista de .npz a procesar
    npz_paths = sorted(glob(os.path.join(npz_input_dir, "*.npz")))
    print(npz_paths)
    if max_imagenes is not None:
        npz_paths = npz_paths[:max_imagenes]

    for npz_path in npz_paths:
        for emotion, vector in emotion_vectors.items():
            multiplicador = multiplicadores.get(emotion, 1.0)

            # Generar nuevo vector y archivo npz
            nuevo_npz = modify_latent_with_emotion(
                npz_path=npz_path,
                emotion=emotion,
                emotion_vectors=emotion_vectors,
                multiplier=multiplicador,
                output_dir=output_npz_dir
            )

            # Nombre para la imagen generada
            image_name = os.path.basename(nuevo_npz).replace(".npz", ".png")

            # Llamar a la función de generación
            generate_one_image_from_npz(
                npz_path=nuevo_npz,
                image_name=image_name,
                outdir_path="/scratch/CelebA/img_emotion_celeba"
            )
