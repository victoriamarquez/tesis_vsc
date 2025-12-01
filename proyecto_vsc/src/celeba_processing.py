import logging
import os
import subprocess
import pandas as pd
import numpy as np
from numpy import load, savez
from pathlib import Path
from glob import glob

from image_processing import generate_one_image_from_npz

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
    logging.info(f"[CelebA] [→] Cargando atributos CelebA desde archivo.")

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
    logging.info("[CelebA] [✔] Atributos CelebA cargados desde archivo.")
    return df

def project_selected_celeba_images_from_df(df, steps):
    """Proyecta un subconjunto de imágenes de CelebA en el espacio latente de StyleGAN2-ADA.

    La función itera sobre los nombres de archivo presentes en la columna 'file_name' 
    del DataFrame de entrada y ejecuta el script `projector.py` de StyleGAN2-ADA 
    para cada imagen. El resultado de la proyección (el vector latente 'w') 
    se guarda como un archivo .npz en el directorio de salida configurado.

    Args:
        df (pandas.DataFrame): DataFrame que contiene, al menos, la columna 
            `"file_name"` con los nombres de las imágenes de CelebA a proyectar.
        steps (int): Número de pasos de optimización a utilizar para la proyección 
            de cada imagen.

    Returns:
        None: La función no devuelve nada, pero genera archivos .npz en el directorio 
            de salida configurado internamente.

    Raises:
        subprocess.CalledProcessError: Si la ejecución del comando de proyección 
            (subprocess.run) falla para alguna de las imágenes.
    """

    logging.info(f"[CelebA] [→] Proyectando {len(df)} imágenes.")
    # Path base en el host
    base_output_host = "/home/vicky/Documents/tesis_vsc/images/CelebA/npz_align_celeba"

    # Path que ve el contenedor
    base_input_docker = "images/CelebA/img_align_celeba"
    base_output_docker = "images/CelebA/npz_align_celeba"

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

        logging.info(f"[CelebA] [→] Proyectando {file_name}.")
        logging.debug(f"[CelebA] Comando: \n{' '.join(command)}")

        subprocess.run(command, check=True)
        logging.info(f"[CelebA] [✔] Proyección de {file_name} finalizada.")

    logging.info(f"[CelebA] [✔] Proyección completa para {len(df)} imágenes de CelebA seleccionadas.")

def modify_latent_with_emotion(npz_path, emotion, emotion_vectors, multiplier, output_dir):
    """
    Aplica una manipulación semántica a un vector latente 'w' sumándole un vector de emoción escalado.

    Toma un archivo .npz que contiene un vector latente 'w' proyectado (shape (1, 18, 512)), 
    le suma el vector diferencial correspondiente a una emoción (broadcasting a través 
    de las 18 capas) y guarda el nuevo vector latente modificado en un nuevo archivo .npz.

    Args:
        npz_path (str): Ruta al archivo .npz original de CelebA.
        emotion (str): Código de la emoción a aplicar, ej. 'HA' (Happy), 'AN' (Angry).
        emotion_vectors (dict): Diccionario que mapea códigos de emoción a sus 
            vectores latentes diferenciales (np.array de shape (512,)).
        multiplier (float): Escalar que multiplica el vector diferencial de emoción, 
            controlando la intensidad del efecto.
        output_dir (str): Carpeta donde guardar el nuevo archivo .npz modificado.

    Returns:
        str: La ruta completa donde se guardó el nuevo archivo .npz modificado.
    """
    # Cargar vector original (w tiene shape (1, 18, 512))
    w = load(npz_path)['w']
    
    # Sumar el vector emocional (broadcast en las 18 capas)
    delta = emotion_vectors[emotion] * multiplier  # (512,)
    delta_expanded = np.tile(delta.to_numpy().reshape(1, 1, 512), (1, 18, 1))  # (1, 18, 512)
    w_modificado = w + delta_expanded

    # Crear nombre nuevo para el archivo
    base_name = Path(npz_path).stem  # ej. "000003_01_projected_w"
    nombre_salida = f"{base_name}_emo_{emotion}.npz"
    path_salida = os.path.join(output_dir, nombre_salida)

    # Guardar nuevo archivo
    os.makedirs(output_dir, exist_ok=True)
    savez(path_salida, w=w_modificado)
    return path_salida  # lo devolvemos para usarlo después con generate_one_image_from_npz

def process_emotions_celeba(emotion_vectors, multiplicadores, max_imagenes=None):
    """
    Aplica vectores emocionales a los NPZ de CelebA, guarda los nuevos NPZ 
    y genera las imágenes resultantes.

    Esta función orquesta el proceso completo de manipulación semántica:
    1. Carga los vectores latentes base de CelebA.
    2. Itera sobre cada emoción y multiplicador.
    3. Llama a `modify_latent_with_emotion` para crear el NPZ modificado.
    4. Llama a `generate_one_image_from_npz` para generar la imagen PNG correspondiente.

    Args:
        npz_input_dir (str): Carpeta con los archivos .npz originales (vectores 'w' proyectados) de CelebA.
        emotion_vectors (dict): Diccionario con vectores emocionales (diferenciales) 
            obtenidos, por ejemplo, por un modelo de regresión.
        multiplicadores (dict): Diccionario que contiene el escalar de intensidad 
            para cada código de emoción. Ej: {'HA': 3.0, 'AN': 4.5, ...}.
        output_npz_dir (str): Carpeta donde guardar los nuevos archivos .npz modificados.
        max_imagenes (int, optional): Si se desea limitar la cantidad de archivos NPZ 
            base procesados para fines de prueba o rendimiento.

    Returns:
        None: La función no devuelve nada, pero genera múltiples archivos .npz y 
            archivos .png de imágenes sintetizadas en sus respectivos directorios de salida.
    """

    # Obtener lista de .npz a procesar
    npz_paths = sorted(glob(os.path.join("/home/vicky/Documents/tesis_vsc/images/CelebA/npz_align_celeba", "*.npz")))
    logging.info(f"[CelebA] [→] Modificando npz y generando imágenes para {len(npz_paths)} archivos.")

    if max_imagenes is not None:
        npz_paths = npz_paths[:max_imagenes]

    for npz_path in npz_paths:
        for emotion, vector in emotion_vectors.items():
            multiplicador = multiplicadores.get(emotion, 1.0)


            logging.debug(f"[CelebA] Llamando modify_latent_with_emotion con parámetros: \n npz_path={npz_path},\n emotion={emotion}, \n emotion_vectors={emotion_vectors},\n multiplier={multiplicador}")
            # Generar nuevo vector y archivo npz
            nuevo_npz = modify_latent_with_emotion(
                npz_path=npz_path,
                emotion=emotion,
                emotion_vectors=emotion_vectors,
                multiplier=multiplicador,
                output_dir="images/CelebA/npz_emotion_celeba"
            )

            logging.debug(f"[CelebA] Llamando generate_one_image_from_npz con parámetros: \n npz_path={nuevo_npz}, \n outdir_path=/scratch/CelebA/img_emotion_celeba")
            # Llamar a la función de generación
            generate_one_image_from_npz(
                npz_path=nuevo_npz,
                outdir_path="/scratch/images/CelebA/img_emotion_celeba"
            )
    logging.info(f"[CelebA] [✔] Generación de imágenes finalizada para {len(npz_paths)} archivos.")

