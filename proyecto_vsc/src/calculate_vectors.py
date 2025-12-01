import numpy as np
import pandas as pd
import logging

from helpers import getNPZ
from image_processing import build_image_dataframe, align_all_images_from_df, generate_all_images, process_all_images
from lineal_regression import compute_emotion_direction_regression
from method_comparison import compare_directions_cosine
from pca import compute_emotion_direction_pca

def calculate_vectors(align=False, process=False, generate=False):
    logging.info("[calculate_vectors] [→] Iniciando pipeline.")

    # Define the base directory
    base_dir = "/home/vicky/Documents/BU_3DFE"

    metadatos_df = build_image_dataframe(base_dir)
    metadatos_df.to_csv("/home/vicky/Documents/tesis_vsc/datos/metadatos.csv", index=False)
    
    if(align):
        logging.info("[calculate_vectors] [→] Alineando imágenes.")
        align_all_images_from_df(
            df=metadatos_df,
            script_path="/home/vicky/Documents/tesis_vsc/stylegan2encoder/align_images.py",
            output_path="/home/vicky/Documents/tesis_vsc/images/aligned_images"
        )
        logging.info("[calculate_vectors] [✔] Imágenes alineadas.")   
    if(process):
        logging.info("[calculate_vectors] [→] Proyectando imágenes a npz.")
        process_all_images('/home/vicky/Documents/tesis_vsc/images/aligned_images', 1000)
        logging.info("[calculate_vectors] [✔] Imágenes proyectadas a npz.")
    if(generate):
        logging.info("[calculate_vectors] [→] Generando imágenes a partir de npz.")
        generate_all_images()
        logging.info("[calculate_vectors] [✔] Imágenes generadas a partir de npz.")
    

    metadatos_df['latent_vector'] = metadatos_df['file_name'].apply(getNPZ)
    metadatos_df.to_csv("/home/vicky/Documents/tesis_vsc/datos/metadatos.csv", index=False)
    metadatos_df.to_pickle("datos/metadatos_con_vectores.pkl")
    metadatos_df = pd.read_pickle("datos/metadatos_con_vectores.pkl")
    
    # ----------------------------------
    # Emociones que vamos a procesar
    # ----------------------------------
    emotion_codes = ['HA', 'AN', 'DI', 'FE', 'SA', 'SU']

    # Diccionarios para guardar resultados
    directions_pca = {}
    directions_regression = {}

    # Suponemos que metadatos_df ya está cargado y tiene la columna 'latent_vector'
    for emotion in emotion_codes:
        try:
            dir_pca = compute_emotion_direction_pca(metadatos_df, emotion)
            directions_pca[emotion] = dir_pca/np.linalg.norm(dir_pca) # NORMALIZAMOS 
            dir_reg = compute_emotion_direction_regression(metadatos_df, emotion)
            directions_regression[emotion] = dir_reg/np.linalg.norm(dir_reg) # NORMALIZAMOS
            
            # Sanity checks
            print(f"PCA norm for {emotion}: {np.linalg.norm(directions_pca[emotion])}")
            print(f"LR norm for {emotion}: {np.linalg.norm(directions_regression[emotion])}")
            print(f"Dot product for {emotion}: {np.dot(directions_pca[emotion], directions_regression[emotion]):.4f}")
            print(f"Cos PCA ↔ LR for {emotion}: {compare_directions_cosine(directions_pca[emotion], directions_regression[emotion]):.4f}")
            print("dir_pca[:10]:", directions_pca[emotion][:10])
            print("dir_reg[:10]:", directions_regression[emotion][:10])


        except ValueError as e:
            print(e)

    # ----------------------------------
    # Comparación de las direcciones: Similaridad del coseno
    # ----------------------------------
    similarities = {}
    for emotion in emotion_codes:
        if emotion in directions_pca and emotion in directions_regression:
            sim = compare_directions_cosine(directions_pca[emotion], directions_regression[emotion])
            similarities[emotion] = sim

    
    pd.DataFrame(directions_regression).to_csv("/home/vicky/Documents/tesis_vsc/datos/directions_regression.csv", index=False)

    # plot_emotion_directions_grid(
    # metadatos_df,
    # directions_pca,
    # directions_regression,
    # similarities,
    # emotion_codes
    # )

    # generate_testing_images_for_multipliers(metadatos_df, directions_pca, directions_regression)

    logging.info("[calculate_vectors] [✔] Ejecución de pipeline finalizada.")

