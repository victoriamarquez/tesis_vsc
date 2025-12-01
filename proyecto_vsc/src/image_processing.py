import logging
import os
import pandas as pd
from helpers import *
from pathlib import Path
import subprocess

# Methods that do the actual processing

def parse_file_name(file_name):
    """Extrae metadatos clave de una cadena de nombre de archivo con formato específico.

    Usada en construir_dataframe_imagenes dentro de calculate_vectors.
    
    El formato esperado es:
    [Género (1)][ID Numérico (4)]_[Emoción (2)][Intensidad (2)][Raza (2)]_[Tipo de Imagen].png

    Ejemplo de nombre de archivo esperado: 'F0001_DI04CE_F2D.png'

    Args:
        file_name (str): El nombre completo del archivo de imagen (e.g., 'F0001_DI04CE_F2D.png').

    Returns:
        dict or None: Un diccionario conteniendo los metadatos extraídos con las 
            siguientes claves:
            * **file_name** (str): El nombre de archivo original.
            * **person_id** (str): ID único de la persona (e.g., 'F0001').
            * **gender** (str): Género ('F' o 'M').
            * **race** (str): Código de la raza (e.g., 'CE').
            * **id_num** (str): Número de identificación (e.g., '0001').
            * **emotion** (str): Código de la emoción (e.g., 'DI').
            * **intensity** (str): Código de la intensidad (e.g., '04').
            * **is_neutral** (bool): True si la emoción es 'NE', False en caso contrario.
            * **img_type** (str): Tipo de imagen (e.g., 'F2D').
        
        Devuelve **None** si ocurre un error (e.g., el nombre del archivo es demasiado corto).

    """
    try:
        gender = file_name[0]  # F o M
        id_num = file_name[1:5]  # '0001'
        person_id = f"{gender}{id_num}"  # 'F0001'
        
        emotion = file_name[6:8]  # 'DI'
        intensity = file_name[8:10]  # '04'
        race = file_name[10:12]

        is_neutral = emotion == "NE"

        img_type = file_name.split("_")[-1].split(".")[0]  # 'F2D'

        return {
            "file_name": file_name,
            "person_id": person_id,
            "gender": gender,
            "race": race,
            "id_num": id_num,
            "emotion": emotion,
            "intensity": intensity,
            "is_neutral": is_neutral,
            "img_type": img_type,
        }
    except Exception as e:
        print(f"Error al parsear {file_name}: {e}")
        return None
    
def build_image_dataframe(ruta_base):
    """Construye un DataFrame de Pandas a partir de metadatos extraídos de nombres de archivos de imágenes.

    Usada en calculate_vectors.
    
    La función recorre recursivamente una estructura de directorios, esperando que 
    los subdirectorios representen a diferentes personas. Itera sobre todas las
    imágenes BMP, extrae metadatos clave utilizando `parse_file_name`, y filtra 
    solo aquellas imágenes cuyo tipo (`img_type`) sea 'F2D'.

    Args:
        ruta_base (str or pathlib.Path): La ruta base del directorio que contiene 
            las carpetas de las personas y las imágenes.

    Returns:
        pandas.DataFrame: Un DataFrame donde cada fila representa una imagen 
            que cumple el criterio 'F2D'. Las columnas incluyen todos los 
            metadatos extraídos por `parse_file_name` más la columna 
            `'original_path'` con la ruta completa del archivo.
            Retorna un DataFrame vacío si no se encuentran imágenes 'F2D' 
            o si no se puede parsear ningún nombre de archivo.

    Notes:
        Requiere la función externa `parse_file_name(file_name)` para decodificar
        los metadatos del nombre de archivo.
    """
    ruta_base = Path(ruta_base)
    data = []

    for carpeta_persona in ruta_base.iterdir():
        if carpeta_persona.is_dir():
            for imagen_path in carpeta_persona.glob("*.bmp"):
                nombre = imagen_path.name
                parsed = parse_file_name(nombre)
                if parsed and parsed["img_type"] == "F2D":  # Filtramos solo imágenes F2D
                    parsed["original_path"] = str(imagen_path)
                    data.append(parsed)

    df = pd.DataFrame(data)
    return df

def align_all_images_from_df(df, script_path, output_path):
    """
    Alinea todas las imágenes usando el script de stylegan2encoder, procesando por carpeta de persona.
    Se basa en el dataframe.
    
    Usada para alinear en calculate_vectors.

    Args:
        df (pd.DataFrame): DataFrame con los metadatos de las imágenes.
        script_path (str or Path): Ruta al script `align_images.py`.
        output_path (str or Path): Carpeta donde se guardarán las imágenes alineadas.
    """
    logging.info(f"[align_all_images_from_df] [→] Alineando imágenes.")

    # Configurar entorno para evitar logs ruidosos
    os.environ['KMP_WARNINGS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    output_path = Path(output_path)
    script_path = Path(script_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Obtener carpetas únicas (una por persona)
    carpetas_persona = df["original_path"].apply(lambda p: Path(p).parent).unique()

    for carpeta in carpetas_persona:
        carpeta = Path(carpeta)
        nombre_persona = carpeta.name
        carpeta_salida = output_path / nombre_persona

        # Evitar reprocesar si ya existe la carpeta alineada
        if carpeta_salida.exists() and any(carpeta_salida.glob("*.png")):
            logging.info(f"[align_all_images_from_df] [✔] Carpeta ya alineada: {nombre_persona}, se saltea.")
            continue

        command = [
            "python3",
            str(script_path),
            str(carpeta),        # Carpeta con imágenes originales de esta persona
            str(output_path),    # Carpeta base de salida
        ]

        logging.info(f"[align_all_images_from_df] [→] Alineando imágenes en: {carpeta}")
        logging.debug(f"[align_all_images_from_df] Comando: \n{' '.join(command)}")

        subprocess.run(command, check=True)
        logging.info(f"[align_all_images_from_df] [✔] Finalizó alineación de imágenes en: {carpeta}")

    logging.info("[align_all_images_from_df] [✔] Alineación completa para todas las carpetas.")

def align_single_image(image_path):
    """
    Alinea UNA imagen usando align_images.py y guarda la imagen alineada
    en su misma carpeta.

    Args:
        image_path (str or Path): Ruta a la imagen original.
        script_path (str or Path): Ruta al script align_images.py.

    Returns:
        Path: Ruta de la imagen alineada generada.
    """

    # Reducir ruido innecesario
    os.environ['KMP_WARNINGS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    image_path = Path(image_path).resolve()
    script_path = Path("/home/vicky/Documents/tesis_vsc/stylegan2encoder/align_images.py").resolve()
    output_dir = image_path.parent.resolve()

    logging.info(f"[align_all_images_from_df] [→] Alineando imagen: {image_path.name}")

    command = [
        "python3",
        str(script_path),
        str(output_dir),  # input_dir absolutizado
        str(output_dir),  # output_dir absolutizado
    ]

    logging.debug(f"[align_all_images_from_df] Comando: \n{' '.join(command)}")
    subprocess.run(command, check=True)

    aligned_candidates = list(output_dir.glob(image_path.stem + "_*.png"))

    if not aligned_candidates:
        raise RuntimeError("❌ No se generó ninguna imagen alineada.")

    aligned_img = aligned_candidates[0]

    logging.info(f"[align_all_images_from_df] [✔] Imagen alineada generada: {aligned_img.name}")
    return aligned_img

def process_all_images(aligned_images_dir, steps):
    
    """Proyecta una colección de imágenes alineadas en el espacio latente de StyleGAN2-ADA.
    Se basa en el directorio.
        
    Usada para generar los npz en calculate_vectors. Se basa en el directorio.
    
    Args:
        aligned_images_dir (str): Ruta al directorio que contiene las imágenes 
            PNG alineadas que se proyectarán.
        steps (int): Número de pasos de optimización a utilizar para la proyección 
            de cada imagen. A mayor número, mejor calidad de proyección, pero más tiempo.

    Returns:
        None: La función no devuelve nada, pero genera las imágenes proyectadas 
            en el directorio de salida configurado internamente.

    Raises:
        subprocess.CalledProcessError: Si la ejecución del comando de proyección 
            (subprocess.run) falla para alguna de las imágenes.
    """
    # Obtener una lista de todas las imágenes en el directorio
    imagenes = [f for f in os.listdir(aligned_images_dir) if f.endswith('.png')]
    logging.info(f"[process_all_images] [→] Proyectando {len(imagenes)} imágenes.")

    # Comando base para ejecutar
    command_base = [
        "/home/vicky/Documents/tesis_vsc/stylegan2-ada-pytorch/docker_run.sh",
        "python", "stylegan2-ada-pytorch/projector.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        f"--num-steps={steps}",
        "--seed=303",
        "--save-video=False"
    ]

    # Ejecutar el comando para cada imagen
    for imagen in imagenes:
        # Cambiar las rutas para que apunten a /scratch en lugar de /mnt/discoAmpliado/viky
        target_path = f"/scratch/images/aligned_images/{imagen}"
        outdir_path = "/scratch/images/processed_images"
        
        # Construir el comando completo
        command = command_base + [
            f"--target={target_path}",
            f"--outdir={outdir_path}"
        ]
        
        # Imprimir el comando para depuración
        logging.info(f"[process_all_images] [→] Proyectando la imagen: {imagen}")
        logging.debug(f"[process_all_images] Comando: \n{' '.join(command)}")
        subprocess.run(command, check=True)
        logging.info(f"[process_all_images] [✔] Proyección finalizada para la imagen: {imagen}")

    logging.info(f"[process_all_images] [✔] Proyección completa para {len(imagenes)} imágenes.")

def project_single_image(image_path, steps=1000):
    """
    Proyecta una sola imagen PNG alineada usando el projector de StyleGAN2-ADA-PyTorch.
    
    Esta función es equivalente a process_all_images(), pero para un único archivo.
    Usa el mismo docker_run.sh y el mismo command_base.
    
    Args:
        image_path (str): Ruta ABSOLUTA a la imagen alineada (ej: /home/.../aligned/myimg_01.png)
        steps (int): Número de pasos de optimización (por defecto 1000, igual que tu pipeline).
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen no existe: {image_path}")

    logging.info(f"[project_single_image] [→] Proyectando imagen única: {image_path}.")

    # Extraer nombre de archivo
    imagen = os.path.basename(image_path)

    target_path = f"/scratch/{str(image_path)}"
    outdir_path = f"/scratch/{image_path.parent}"

    # Comando base (idéntico al original)
    command = [
        "/home/vicky/Documents/tesis_vsc/stylegan2-ada-pytorch/docker_run.sh",
        "python", "stylegan2-ada-pytorch/projector.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        f"--num-steps={steps}",
        "--seed=303",
        "--save-video=False",
        f"--target={target_path}",
        f"--outdir={outdir_path}"
    ]

    logging.debug(f"[project_single_image] Comando: \n{' '.join(command)}")
    subprocess.run(command, check=True)
    logging.info(f"[project_single_image] [✔] Proyección completada: {imagen}")

def generate_one_image_from_npz(npz_path, outdir_path):
    """Genera una imagen a partir de un archivo NPZ que contiene un vector latente 'w' proyectado.

    Usada en generate_modified_emotion_images dentro de diverse_group_testing y en process_emotions_celeba dentro de celeba_processing.
    
    La función toma la ruta a un archivo NPZ (que típicamente contiene los pesos 'w' 
    obtenidos de una proyección StyleGAN), un nombre de archivo para la imagen de salida, 
    y el directorio de destino. Ejecuta el script `generate.py` de StyleGAN2-ADA 
    a través de un script de Docker para generar la imagen final.

    Args:
        npz_path (str or pathlib.Path): Ruta completa al archivo NPZ que contiene 
            el vector latente proyectado (e.g., 'w_proyectado.npz').
        image_name (str): El nombre que se asignará a la imagen generada.
            (Nota: El script StyleGAN puede añadir sufijos o extensiones).
        outdir_path (str or pathlib.Path): Directorio de salida donde se guardará 
            la imagen generada.

    Returns:
        None: La función no devuelve un valor, pero genera el archivo de imagen 
            en el directorio especificado.

    Raises:
        subprocess.CalledProcessError: Si la ejecución del comando de generación 
            (subprocess.run) falla.
    """

    command = [
        "/home/vicky/Documents/tesis_vsc/stylegan2-ada-pytorch/docker_run.sh",
        "python",
        "stylegan2-ada-pytorch/generate.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        f"--outdir={outdir_path}",
        f"--projected-w={npz_path}"
    ]    
    logging.info(f"[generate_one_image_from_npz] [→] Generando imagen para el npz: {npz_path}.")
    logging.debug(f"[generate_one_image_from_npz] Comando: \n{' '.join(command)}")
    subprocess.run(command, check=True)
    logging.info(f"[generate_one_image_from_npz] [✔] Imagen generada para el npz: {npz_path}.")

def generate_all_images():
    """Genera imágenes a partir de todos los archivos NPZ de vectores latentes proyectados.

    Usada para generar las imágenes en calculate_vectors. Se basa en el directorio.

    La función escanea un directorio predefinido (`processed_images_dir`) para encontrar 
    todos los archivos `.npz` que contienen los vectores latentes 'w' proyectados (obtenidos
    previamente por la función de proyección). Luego, itera sobre cada archivo NPZ 
    y ejecuta el script `generate.py` de StyleGAN2-ADA para sintetizar la imagen 
    correspondiente a ese vector latente.

    Returns:
        None: La función no devuelve un valor, pero guarda todas las imágenes generadas 
            en el directorio de salida configurado internamente (`/scratch/images/generated_images`).

    Raises:
        subprocess.CalledProcessError: Si la ejecución del comando de generación 
            (subprocess.run) falla para alguno de los archivos NPZ.
    """
    
    processed_images_dir = '/home/vicky/Documents/tesis_vsc/images/processed_images'

    # Obtener una lista de todas las imágenes en el directorio
    imagenes = [f for f in os.listdir(processed_images_dir) if f.endswith('.npz')]
    logging.info(f"[generate_all_images] [→] Generando {len(imagenes)} imágenes.")

    command_base = ["/home/vicky/Documents/tesis_vsc/stylegan2-ada-pytorch/docker_run.sh",
        "python",
        "stylegan2-ada-pytorch/generate.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    ]

    for imagen in imagenes:
        # Cambiar las rutas para que apunten a /scratch en lugar de /mnt/discoAmpliado/viky
        npz_path = f"/scratch/images/processed_images/{imagen}"
        outdir_path = "/scratch/images/generated_images"
        # Construir el comando completo
        command = command_base + [
            f"--outdir={outdir_path}",
            f"--projected-w={npz_path}"
        ]

        # Imprimir el comando para depuración
        logging.info(f"[generate_all_images] [→] Generando imagen: {imagen}")
        logging.debug(f"[generate_all_images] Comando: \n{' '.join(command)}")
        subprocess.run(command, check=True)
        logging.info(f"[generate_all_images] [✔] Finalizó generación de imagen imagen: {imagen}")

    logging.info(f"[generate_all_images] [✔] Generación completa para {len(imagenes)} imágenes.")
    