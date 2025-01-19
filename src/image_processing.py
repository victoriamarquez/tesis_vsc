import os
import pandas as pd
from helpers import *
import torch
import gc

# Methods that do the actual processing


def align_images(images_data, verbosity=True):
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
            'python3', '/home/vicky/Documents/tesis/stylegan2encoder/align_images.py',
            folder,  # Carpeta actual a procesar
            aligned_image_path,
        ]

        optional_print(f"Running command for folder: {folder}", verbosity)
        optional_print("Command: " + " ".join(command), verbosity)
        
        # Ejecutar el comando
        subprocess.run(command, check=True)

    optional_print("Processing complete.", verbosity)



def batch_processing_legacy(images_data, verbosity=True):
    # Paths
    aligned_image_path = '/mnt/discoAmpliado/viky/images/aligned_images'
    model_script_path = '/home/vicky/Documents/tesis/stylegan2-ada-pytorch/docker_run.sh'
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
            
            target_image_path = os.path.join(aligned_image_path, image['file'])

            #command = f"{model_path} python3 ./stylegan2-ada-pytorch/projector.py --outdir={base_outdir}/{image_name} --target={aligned_image_path}/{image_name}_01.png --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --num-steps=500"
            command = [
                'bash', model_script_path,
                'python3', '/home/vicky/Documents/tesis/stylegan2-ada-pytorch/projector.py',
                f'--outdir={base_outdir}/{image_name}',
                f'--target={aligned_image_path}/{image_name}_01.png',
                '--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl',
                '--num-steps=500'
            ]

            optional_print(f"Running command for image: {image['file']}", verbosity)
            optional_print("Command: " + " ".join(command), verbosity)
            
            # Run the command
            #run_command(command)

            try:
                subprocess.run(command, check=True)
                image['projected_file'] = f'{base_outdir}/{image_name}/projected_w.npz'
            except subprocess.CalledProcessError as e:
                optional_print(f"Error processing image {image['file']}: {e}", verbosity)

            #image['projected_npz'] = cargar_npz(f'BU_3DFE/{image_name}')
            image['projected_file'] = f'{base_outdir}/{image_name}/projected_w.npz'
        
        # Save the current batch to a CSV file
        df_batch = pd.DataFrame(batch)
        df_batch.to_csv(f'/mnt/discoAmpliado/viky/dataframes/processed_dataframe_batch_{batch_num + 1}.csv', index=False)
        
        optional_print(f"Batch {batch_num + 1}/{total_batches} processed and saved.", verbosity)
        optional_print(f"{end_index} out of {total_images} images processed so far.", verbosity)

    optional_print("All specified batches processed.", verbosity)

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


