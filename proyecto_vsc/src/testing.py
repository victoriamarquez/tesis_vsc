import pandas as pd

from diverse_group_testing import generate_diverse_testing_subset, generate_modified_emotion_images
from celeba_processing import load_celeba_attributes, process_emotions_celeba, process_selected_celeba_images_from_df


def execute_tests():
    metadatos_df = pd.read_csv("/home/vicky/Documents/tesis_vsc/datos/metadatos.csv")
    directions_regression = pd.read_csv("/home/vicky/Documents/tesis_vsc/datos/directions_regression.csv")
    diverse_subset = generate_diverse_testing_subset(metadatos_df, False)

    generate_modified_emotion_images(
        subset_df=diverse_subset,
        directions_dict=directions_regression,
        emotion_multipliers={  ##### AJUSTAR ESTO
            'HA': 5,
            'AN': 2,
            'DI': 0.5,
            'FE': 2,
            'SA': 4,
            'SU': 4.5
        },
        method_name="LR"
    )

    celeba_df = load_celeba_attributes("/home/vicky/Documents/tesis_vsc/list_attr_celeba.txt")
    celeba_neutral_df = celeba_df[(celeba_df['Smiling'] == -1) & (celeba_df['Eyeglasses'] == -1)]
    celeba_sample_df = celeba_neutral_df.sample(n=10, random_state=42) # Elegimos 10 imágenes al azar para probar
    process_selected_celeba_images_from_df(celeba_sample_df, 1000, True)
    multiplicadores = {
        'HA': 5,
        'AN': 2,
        'DI': 0.5,
        'FE': 2,
        'SA': 4,
        'SU': 4.5
    }

    process_emotions_celeba(
        npz_input_dir="/home/vicky/Documents/tesis_vsc/images/CelebA/npz_aligned_celeba", ## ESTO ESTÁ VACÍO
        emotion_vectors=directions_regression,
        multiplicadores=multiplicadores,
        output_npz_dir="CelebA/npz_emotion_celeba",
        max_imagenes=10  # opcional para probar con pocas imágenes
    )
