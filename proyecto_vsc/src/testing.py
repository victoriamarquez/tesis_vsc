import logging
import pandas as pd

from diverse_group_testing import generate_diverse_testing_subset, generate_modified_emotion_images
from celeba_processing import load_celeba_attributes, process_emotions_celeba, project_selected_celeba_images_from_df
from helpers import getNPZ

multiplicadores = {
        'HA': 5,
        'AN': 2,
        'DI': 0.5,
        'FE': 2,
        'SA': 4,
        'SU': 4.5
    }

def execute_tests():
    """
    Ejecuta un conjunto predefinido de pruebas de manipulación emocional sobre 
    diferentes datasets (propio y CelebA).

    La función carga los metadatos de las imágenes proyectadas, incluyendo los 
    vectores latentes, y los vectores de dirección emocional obtenidos por 
    Regresión Lineal (`directions_regression`). Luego, puede ejecutar dos bloques 
    de pruebas:

    1. **Prueba de Diversidad:** Genera un subconjunto diverso de imágenes neutras
        y aplica las direcciones emocionales sobre ellas.
    2. **Prueba de CelebA:** Procesa un subconjunto de imágenes neutras de CelebA, 
       realiza la proyección inicial de esas imágenes y luego aplica las direcciones 
       emocionales calculadas para sintetizar nuevas imágenes.

    Returns:
        None: La función no devuelve un valor, pero genera numerosos archivos .npz 
            (vectores latentes modificados) y archivos .png (imágenes sintetizadas) 
            en directorios de salida predefinidos.

    Notes:
        Depende de funciones auxiliares como `getNPZ`, `generate_diverse_testing_subset`, 
        `generate_modified_emotion_images`, `load_celeba_attributes`, 
        `process_selected_celeba_images_from_df` y `process_emotions_celeba`.
    """

    metadatos_df = pd.read_csv("/home/vicky/Documents/tesis_vsc/datos/metadatos.csv")
    metadatos_df['latent_vector'] = metadatos_df['file_name'].apply(getNPZ)
    
    directions_regression = pd.read_csv("/home/vicky/Documents/tesis_vsc/datos/directions_regression.csv")
    
    logging.info("[execute_tests] [→] Iniciando pruebas subset diversidad.")
    diverse_subset = generate_diverse_testing_subset(metadatos_df)

    generate_modified_emotion_images(
        subset_df=diverse_subset,
        directions_dict=directions_regression,
        emotion_multipliers=multiplicadores,
        method_name="LR"
    )
    logging.info("[execute_tests] [✔] Pruebas subset diversidad finalizadas.")

    logging.info("[execute_tests] [→] Iniciando pruebas con dataset CelebA.")
    celeba_df = load_celeba_attributes("/home/vicky/Documents/tesis_vsc/list_attr_celeba.txt")
    celeba_neutral_df = celeba_df[(celeba_df['Smiling'] == -1) & (celeba_df['Eyeglasses'] == -1) & (celeba_df['Blurry'] == -1)]
    celeba_sample_df = celeba_neutral_df.sample(n=10, random_state=42) # Elegimos 10 imágenes al azar para probar
    project_selected_celeba_images_from_df(celeba_sample_df, 1000)

    process_emotions_celeba(
        emotion_vectors=directions_regression,
        multiplicadores=multiplicadores,
        max_imagenes=10  # opcional para probar con pocas imágenes
    )
    logging.info("[execute_tests] [✔] Pruebas con dataset CelebA finalizadas.")
