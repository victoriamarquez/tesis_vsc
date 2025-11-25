import logging
import os
import pandas as pd
from helpers import *
import torch
import gc
from pathlib import Path

# Methods that do the actual processing

def align_all_images_from_df(df, script_path, output_path, verbose=True):
    """
    Alinea todas las imágenes usando el script de stylegan2encoder, procesando por carpeta de persona.

    Args:
        df (pd.DataFrame): DataFrame con los metadatos de las imágenes.
        script_path (str or Path): Ruta al script `align_images.py`.
        output_path (str or Path): Carpeta donde se guardarán las imágenes alineadas.
        verbose (bool): Si True, imprime el progreso.
    """
    logging.debug(f"Función: align_all_images_from_df. Argumentos: {df.head()}, {script_path}, {output_path}.")
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
            if verbose:
                print(f"[✔] Carpeta ya alineada: {nombre_persona}, se salta.")
            continue

        command = [
            "python3",
            str(script_path),
            str(carpeta),        # Carpeta con imágenes originales de esta persona
            str(output_path),    # Carpeta base de salida
        ]

        if verbose:
            logging.info(f"[→] Alineando imágenes en: {carpeta}")
            logging.info(f"    Comando: {' '.join(command)}")

        subprocess.run(command, check=True)

    if verbose:
        print("\n[✔] Alineación completa para todas las carpetas.")

##### LEGACY

def align_all_images(images_data, verbosity=True):
    # Suppress OpenMP and TensorFlow logs
    os.environ['KMP_WARNINGS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Obtener carpetas únicas
    unique_folders = {image['raw_image_folder'] for image in images_data}

    aligned_image_path = '/mnt/discoAmpliado/viky/images/aligned_images'

    # Crear el directorio de salida si no existe
    os.makedirs(aligned_image_path, exist_ok=True)

    for folder in unique_folders:
        command = [
            'python3', '/mnt/discoAmpliado/viky/stylegan2encoder/align_images.py',
            folder,  # Carpeta actual a procesar
            aligned_image_path,
        ]

        optional_print(f"Running command for folder: {folder}", verbosity)
        optional_print("Command: " + " ".join(command), verbosity)
        
        # Ejecutar el comando
        subprocess.run(command, check=True)

    optional_print("Alineación completa para todas las imágenes.", verbosity)

def batch_processing(images_data, verbosity=True):
    # Paths
    aligned_image_path = '/mnt/discoAmpliado/viky/images/aligned_images'
    base_outdir = '/mnt/discoAmpliado/viky/images/processed_images'
    
    # Define the batch size
    batch_size = 10

    # Calculate the total number of batches
    total_images = len(images_data)
    total_batches = total_images // batch_size + (1 if total_images % batch_size != 0 else 0)

    # Specify the start and end batches (inclusive)
    start_batch = 0  # Modify this to your desired starting batch number
    end_batch = total_batches - 1  # Modify this to your desired ending batch number

    # Ensure the end batch is within the valid range
    end_batch = min(end_batch, total_batches - 1)

    # Iterate over the specified batches
    for batch_num in range(start_batch, end_batch + 1):
        start_index = batch_num * batch_size
        end_index = min(start_index + batch_size, total_images)
        batch = images_data[start_index:end_index]
        
        # Process each image in the current batch
        for image in batch:
            image_name = os.path.splitext(image['file'])[0]
            
            # Paths relative to the current directory (pwd)
            relative_target_path = os.path.relpath(
                f"{aligned_image_path}/{image_name}_01.png",
                os.getcwd()
            )
            relative_outdir_path = os.path.relpath(
                f"{base_outdir}/{image_name}",
                os.getcwd()
            )

            workdir = "/mnt/discoAmpliado/viky/stylegan2-ada-pytorch"

            relative_target_path = os.path.relpath(f"{aligned_image_path}/{image_name}_01.png", "/mnt/discoAmpliado/viky/stylegan2-ada-pytorch")
            relative_outdir_path = os.path.relpath(f"{base_outdir}/{image_name}", "/mnt/discoAmpliado/viky/stylegan2-ada-pytorch")

            command = [
                'bash', './docker_run.sh',
                'python3', './projector.py',
                f'--outdir={relative_outdir_path}',
                f'--target={relative_target_path}',
                '--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl',
                '--num-steps=500'
            ]

            optional_print(f"Running command for image: {image['file']}", verbosity)
            optional_print("Command: " + " ".join(command), verbosity)
            
            try:
                result = subprocess.run(command, cwd=workdir, check=True, capture_output=True, text=True)
                image['projected_file'] = f'{base_outdir}/{image_name}/projected_w.npz'
                print("Command output:", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error executing command:", e.stderr)
                print("Command failed with return code:", e.returncode)
                print("Command output:", e.output)
                print("Full command:", " ".join(command))

        # Save the current batch to a CSV file
        df_batch = pd.DataFrame(batch)
        df_batch.to_csv(f'/mnt/discoAmpliado/viky/dataframes/processed_dataframe_batch_{batch_num + 1}.csv', index=False)
        
        optional_print(f"Batch {batch_num + 1}/{total_batches} processed and saved.", verbosity)
        optional_print(f"{end_index} out of {total_images} images processed so far.", verbosity)

    optional_print("All specified batches processed.", verbosity)
    return combine_dataframes(total_batches, verbosity)

def process_one_image(image_name, steps, verbose=False):

    # Comando base para ejecutar
    command_base = [
        "/mnt/discoAmpliado/viky/stylegan2-ada-pytorch/docker_run.sh",
        "python", "stylegan2-ada-pytorch/projector.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        f"--num-steps={steps}",
        "--seed=303",
        "--save-video=False"
    ]

    # Cambiar las rutas para que apunten a /scratch en lugar de /mnt/discoAmpliado/viky
    target_path = f"/scratch/images/aligned_images/{image_name}_01.png"
    outdir_path = "/scratch/images/processed_images"
    
    # Construir el comando completo
    command = command_base + [
        f"--target={target_path}",
        f"--outdir={outdir_path}"
    ]
    
    # Imprimir el comando para depuración
    optional_print(f"Ejecutando para la imagen: {image_name}", verbose)
    optional_print(f"Comando: {' '.join(command)}", verbose)

    # Ejecutar el comando
    subprocess.run(command, check=True)

    optional_print("Proyección completa para todas las imágenes.", verbose)


def process_all_images(aligned_images_dir, steps, verbose=False):
    logging.debug(f"Función: process_all_images. Argumentos {aligned_images_dir}, {steps}.")
    # Directorio donde están las imágenes

    # Obtener una lista de todas las imágenes en el directorio
    imagenes = [f for f in os.listdir(aligned_images_dir) if f.endswith('.png')]

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
        logging.info(f"Ejecutando para la imagen: {imagen}")
        logging.info(f"Comando: {' '.join(command)}")

        # Ejecutar el comando
        subprocess.run(command, check=True)

    logging.info("Proyección completa para todas las imágenes.")

def process_all_images_in_df(df, steps, verbose=True):
    # Directorio donde están las imágenes

    # Obtener una lista de todas las imágenes en el directorio
    ##imagenes = [f for f in os.listdir(aligned_images_dir) if f.endswith('.png')]

    imagenes = [f for f in df['name']]

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
    for index, imagen in enumerate(imagenes):
        optional_print(f"Iniciando proyección para imagen {index} de {len(imagenes)}", verbose)
        # Cambiar las rutas para que apunten a /scratch en lugar de /mnt/discoAmpliado/viky
        target_path = f"/scratch/images/aligned_images/{imagen}_01.png"
        outdir_path = "/scratch/images/processed_images"
        
        # Construir el comando completo
        command = command_base + [
            f"--target={target_path}",
            f"--outdir={outdir_path}"
        ]
        
        # Imprimir el comando para depuración
        optional_print(f"Ejecutando para la imagen: {imagen}", verbose)
        optional_print(f"Comando: {' '.join(command)}", verbose)

        # Ejecutar el comando
        subprocess.run(command, check=True)
        optional_print(f"Proyección completa para imagen {index} de {len(imagenes)}", verbose)

    optional_print("Proyección completa para todas las imágenes.", verbose)


def generate_one_image(image_name, verbose=False):

    npz_path = f"/scratch/images/processed_images/{image_name}_01_projected_w.npz"
    outdir_path = "/scratch/images/generated_images"
    
    command = ["/mnt/discoAmpliado/viky/stylegan2-ada-pytorch/docker_run.sh",
        "python",
        "stylegan2-ada-pytorch/generate.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        f"--outdir={outdir_path}",
        f"--projected-w={npz_path}"
    ]    
    optional_print(f"Ejecutando para la imagen: {image_name}", verbose)
    optional_print(f"Comando: {' '.join(command)}", verbose)
    subprocess.run(command, check=True)

def generate_one_image_from_npz(npz_path, image_name, outdir_path):
    ###outdir_path = "/scratch/images/generated_prueba_diversidad"
    
    command = [
        "/home/vicky/Documents/tesis_vsc/stylegan2-ada-pytorch/docker_run.sh",
        "python",
        "stylegan2-ada-pytorch/generate.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        f"--outdir={outdir_path}",
        f"--projected-w={npz_path}"
    ]    
    print(f"Ejecutando para la imagen: {image_name}")
    print(f"Comando: {' '.join(command)}")
    subprocess.run(command, check=True)


def generate_all_images(verbose=False):
    logging.debug("Función: generate_all_images.")

    processed_images_dir = '/home/vicky/Documents/tesis_vsc/images/processed_images'

    # Obtener una lista de todas las imágenes en el directorio
    imagenes = [f for f in os.listdir(processed_images_dir) if f.endswith('.npz')]

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
        logging.info(f"Ejecutando para la imagen: {imagen}")
        logging.info(f"Comando: {' '.join(command)}")
        subprocess.run(command, check=True)

    logging.info("Generación completa para todas las imágenes.")
    