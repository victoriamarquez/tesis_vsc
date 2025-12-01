import subprocess
import os

# Directorio donde están las imágenes
aligned_images_dir = '/mnt/discoAmpliado/viky/images/aligned_images/'

# Obtener una lista de todas las imágenes en el directorio
imagenes = [f for f in os.listdir(aligned_images_dir) if f.endswith('.png')]

# Comando base para ejecutar
command_base = [
    "/mnt/discoAmpliado/viky/stylegan2-ada-pytorch/docker_run.sh",
    "python", "stylegan2-ada-pytorch/projector.py",
    "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
    "--num-steps=1000",
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
    print(f"Ejecutando para la imagen: {imagen}")
    print(f"Comando: {' '.join(command)}")  # Esto te ayudará a ver el comando completo

    # Ejecutar el comando
    subprocess.run(command, check=True)

print("Completado para todas las imágenes.")
