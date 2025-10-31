
import gc
import os
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as  sns

from helpers import getNPZ, optional_print, save_modified_npz
from image_pre_processing import construir_dataframe_imagenes
from image_processing import align_all_images_from_df, generate_all_images, process_all_images
from celebA_processing import load_celeba_attributes, process_emotions_celeba, process_selected_celeba_images_from_df
from diverse_group_testing import generate_diverse_testing_subset, generate_modified_emotion_images
from lineal_regression import compute_emotion_direction_regression
from method_comparison import compare_directions_cosine
from pca import compute_emotion_direction_pca

def main(align=False, process=False, generate=False, diverse_test=False, celebA=False, verbose=True):
    optional_print("_____________[INICIANDO EJECUCIÓN]_____________", verbose)

    gc.collect()
    torch.cuda.empty_cache()

    # Define the base directory
    base_dir = "/mnt/discoAmpliado/viky/BU_3DFE"
    ##### TODO: Hay que cambiar este directorio

    metadatos_df = construir_dataframe_imagenes(base_dir)
    metadatos_df.to_csv("datos/metadatos.csv", index=False)
    
    if(align):
        optional_print("_____________[ALINEANDO]_____________", verbose)
        align_all_images_from_df(
            df=metadatos_df,
            script_path="/mnt/discoAmpliado/viky/stylegan2encoder/align_images.py",
            output_path="/mnt/discoAmpliado/viky/images/aligned_images",
            verbose=True
        )
        optional_print("_____________[ALINEADO]_____________", verbose)   
    if(process):
        optional_print("_____________[PROCESANDO]_____________", verbose)
        #process_all_images_from_df(metadatos_df, steps=200, verbose=True, max_images=3)
        process_all_images('/mnt/discoAmpliado/viky/images/aligned_images/', 1000, verbose) ## Vamos a hacer de cuenta que esto está bien porque no sé cómo arreglarlo
        optional_print("_____________[PROCESADO]_____________", verbose)
    if(generate):
        optional_print("_____________[GENERANDO]_____________", verbose)
        generate_all_images(verbose)
        optional_print("_____________[GENERADO]_____________", verbose)
    
    ##optional_print(f"_____________[ALINEADAS CHECK] {check_alineadas(metadatos_df, False)}_____________", verbose)
    ##optional_print(f"_____________[NPZ CHECK] {check_npz(metadatos_df, False)[0]}_____________", verbose)
    metadatos_df['latent_vector'] = metadatos_df['file_name'].apply(getNPZ)
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
            #print(f"PCA norm for {emotion}: {np.linalg.norm(directions_pca[emotion])}")
            #print(f"LR norm for {emotion}: {np.linalg.norm(directions_regression[emotion])}")
            #print(f"Dot product for {emotion}: {np.dot(directions_pca[emotion], directions_regression[emotion]):.4f}")
            #print(f"Cos PCA ↔ LR for {emotion}: {compare_directions_cosine(directions_pca[emotion], directions_regression[emotion]):.4f}")
            #print("dir_pca[:10]:", directions_pca[emotion][:10])
            #print("dir_reg[:10]:", directions_regression[emotion][:10])


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
            # print(f"Vector para la emoción {emotion} obtenido con PCA: {directions_pca[emotion]}")
            # print(f"Vector para la emoción {emotion} obtenido con RL: {directions_regression[emotion]}")

    # plot_emotion_directions_grid(
    # metadatos_df,
    # directions_pca,
    # directions_regression,
    # similarities,
    # emotion_codes
    # )

    # generate_testing_images_for_multipliers(metadatos_df, directions_pca, directions_regression)

    if diverse_test:
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

    if celebA:
        celeba_df = load_celeba_attributes("/mnt/discoAmpliado/viky/CelebA/list_attr_celeba.txt")
        celeba_neutral_df = celeba_df[(celeba_df['Smiling'] == -1) & (celeba_df['Eyeglasses'] == -1)]
        celeba_sample_df = celeba_neutral_df.sample(n=10, random_state=42) # Elegimos 10 imágenes al azar para probar
        #######process_selected_celeba_images_from_df(celeba_sample_df, 1000, True)
        multiplicadores = {
            'HA': 5,
            'AN': 2,
            'DI': 0.5,
            'FE': 2,
            'SA': 4,
            'SU': 4.5
        }

        process_emotions_celeba(
            npz_input_dir="/mnt/discoAmpliado/viky/CelebA/npz_aligned_celeba",
            emotion_vectors=directions_regression,
            multiplicadores=multiplicadores,
            output_npz_dir="CelebA/npz_emotion_celeba",
            max_imagenes=10  # opcional para probar con pocas imágenes
        )


    optional_print("_____________[EJECUCIÓN FINALIZADA]_____________", verbose)

main(align=True, process=True, generate=True, diverse_test=True, celebA=True, verbose=True)