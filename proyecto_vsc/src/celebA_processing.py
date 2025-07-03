import os
import subprocess

import pandas as pd


def load_celeba_attributes(attr_path):
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

def process_celeba_images(df, steps=300, verbose=True):
    # Ruta a las imágenes alineadas de CelebA
    aligned_images_dir = "/mnt/discoAmpliado/viky/CelebA/img_align_celeba"
    
    # Ruta de salida para los vectores .npz
    outdir_path = "/scratch/images/celeba_processed"

    command_base = [
        "/mnt/discoAmpliado/viky/stylegan2-ada-pytorch/docker_run.sh",
        "python", "stylegan2-ada-pytorch/projector.py",
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
        f"--num-steps={steps}",
        "--seed=303",
        "--save-video=False"
    ]

    for _, row in df.iterrows():
        image_name = row["file_name"]
        target_path = os.path.join("/scratch/images/celeba_aligned", image_name)
        
        command = command_base + [
            f"--target={target_path}",
            f"--outdir={outdir_path}"
        ]
        
        if verbose:
            print(f"Ejecutando proyección para {image_name}")
            print("Comando:", " ".join(command))
        
        subprocess.run(command, check=True)

    print("✔ Proyección completada para CelebA.")