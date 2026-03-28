import logging
import os
import pandas as pd

from diverse_group_testing import generate_diverse_testing_subset, generate_modified_emotion_images
from celeba_helpers import align_celeba_candidates, load_celeba_attributes, replace_background_with_black
from celeba_processing import process_emotions_celeba, project_selected_celeba_images_from_df
from helpers import getNPZ

multiplicadores = {
        'HA': 5,
        'AN': 2,
        'DI': 0.5,
        'FE': 2,
        'SA': 4,
        'SU': 4.5
    }

def execute_tests(celeba_only=False, black_bg=False):
    """
    Ejecuta un conjunto predefinido de pruebas de manipulación emocional sobre 
    diferentes datasets (propio y CelebA).

    Args:
        celeba_only (bool): Si es True, omite las pruebas del subset diverso y
            ejecuta solo la parte de CelebA. Útil para iterar más rápido sobre
            el experimento de CelebA sin esperar el bloque de diversidad.
        black_bg (bool): Si es True, reemplaza el fondo de las imágenes alineadas
            con negro sólido antes de proyectar. Los resultados se guardan en
            carpetas distintas para poder comparar con la ejecución sin black_bg.
    """

    directions_regression = pd.read_csv("/home/vicky/Documents/tesis_vsc/datos/directions_regression.csv")

    if not celeba_only:
        metadatos_df = pd.read_csv("/home/vicky/Documents/tesis_vsc/datos/metadatos.csv")
        metadatos_df['latent_vector'] = metadatos_df['file_name'].apply(getNPZ)

        logging.info("[execute_tests] [→] Iniciando pruebas subset diversidad.")
        diverse_subset = generate_diverse_testing_subset(metadatos_df)
        generate_modified_emotion_images(
            subset_df=diverse_subset,
            directions_dict=directions_regression,
            emotion_multipliers=multiplicadores,
            method_name="LR"
        )
        logging.info("[execute_tests] [✔] Pruebas subset diversidad finalizadas.")

    logging.info("[execute_tests] [→] Iniciando pruebas con dataset CelebA (candidatas alineadas).")

    candidatas_dir = "/home/vicky/Documents/tesis_vsc/images/CelebA/candidatas_revision"
    aligned_dir = "/home/vicky/Documents/tesis_vsc/images/CelebA/candidatas_aligned"

    align_celeba_candidates(candidatas_dir, aligned_dir)

    if black_bg:
        black_bg_dir = "/home/vicky/Documents/tesis_vsc/images/CelebA/candidatas_aligned_black_bg"
        replace_background_with_black(aligned_dir, black_bg_dir)
        project_input_dir = black_bg_dir
        project_input_docker = "images/CelebA/candidatas_aligned_black_bg"
        npz_dir_host = "/home/vicky/Documents/tesis_vsc/images/CelebA/npz_candidatas_black_bg"
        npz_dir_docker = "images/CelebA/npz_candidatas_black_bg"
        output_npz_dir = "images/CelebA/npz_emotion_candidatas_black_bg"
        output_img_dir = "/scratch/images/CelebA/img_emotion_candidatas_black_bg"
    else:
        project_input_dir = aligned_dir
        project_input_docker = "images/CelebA/candidatas_aligned"
        npz_dir_host = "/home/vicky/Documents/tesis_vsc/images/CelebA/npz_candidatas_aligned"
        npz_dir_docker = "images/CelebA/npz_candidatas_aligned"
        output_npz_dir = "images/CelebA/npz_emotion_candidatas"
        output_img_dir = "/scratch/images/CelebA/img_emotion_candidatas"

    # Construir dataframe con las imágenes del directorio de entrada a proyectar
    input_filenames = [f for f in os.listdir(project_input_dir) if f.endswith(".png")]
    input_df = pd.DataFrame({"file_name": input_filenames})

    project_selected_celeba_images_from_df(
        df=input_df,
        steps=1000,
        input_dir_docker=project_input_docker,
        output_dir_host=npz_dir_host,
        output_dir_docker=npz_dir_docker,
    )

    process_emotions_celeba(
        emotion_vectors=directions_regression,
        multiplicadores=multiplicadores,
        npz_input_dir=npz_dir_host,
        output_npz_dir=output_npz_dir,
        output_img_dir=output_img_dir,
    )
    logging.info("[execute_tests] [✔] Pruebas con dataset CelebA finalizadas.")
