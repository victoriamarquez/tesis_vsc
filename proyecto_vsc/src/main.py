
import gc
import os
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as  sns

from helpers import getNPZ, optional_print, save_modified_npz
from image_pre_processing import construir_dataframe_imagenes
from image_processing import align_all_images_from_df, generate_all_images, generate_one_image_from_npz, process_all_images
from celebA_processing import load_celeba_attributes

def plot_emotion_directions_grid(df, directions_pca, directions_regression, similarities, emotion_codes):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, emotion in enumerate(emotion_codes):
        ax = axes[idx]

        # --- Obtener diffs neutro - emocion ---
        diffs = []
        grouped = df[df['emotion'] == emotion].groupby('person_id')
        for person_id, group in grouped:
            neutral_row = df[(df['person_id'] == person_id) & (df['is_neutral'])]
            emo_row = group[group['intensity'] == '04']
            if len(neutral_row) == 1 and len(emo_row) == 1:
                neutral_vec = neutral_row.iloc[0]['latent_vector']
                emo_vec = emo_row.iloc[0]['latent_vector']
                diffs.append(emo_vec - neutral_vec)

        if len(diffs) == 0:
            ax.set_title(f"{emotion}: sin datos")
            continue

        diffs = np.stack(diffs)
        diffs_centered = diffs - diffs.mean(axis=0)

        # --- Proyección PCA a 2D ---
        from sklearn.decomposition import PCA as SKPCA
        pca_2d = SKPCA(n_components=2)
        diffs_2d = pca_2d.fit_transform(diffs_centered)

        # --- Proyectar direcciones al mismo espacio 2D ---
        dir_pca = directions_pca[emotion]
        dir_reg = directions_regression[emotion]

        dir_pca_2d = pca_2d.transform([dir_pca - diffs.mean(axis=0)])[0]
        dir_reg_2d = pca_2d.transform([dir_reg - diffs.mean(axis=0)])[0]

        # --- Normalizar flechas para visualización ---
        def normalize(v): return v / np.linalg.norm(v)
        scale = 5  # alargar flechas para que se vean bien
        dir_pca_2d = normalize(dir_pca_2d) * scale
        dir_reg_2d = normalize(dir_reg_2d) * scale

        # --- Gráfico ---
        ax.scatter(diffs_2d[:, 0], diffs_2d[:, 1], alpha=0.4, s=10)
        ax.quiver(0, 0, dir_pca_2d[0], dir_pca_2d[1], color='blue', scale=1, scale_units='xy', angles='xy', label='PCA')
        ax.quiver(0, 0, dir_reg_2d[0], dir_reg_2d[1], color='orange', scale=1, scale_units='xy', angles='xy', label='Regresión')

        sim_val = similarities.get(emotion, 0)
        ax.set_title(f"{emotion} — coseno: {sim_val:.2f}")
        ax.legend()
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("datos/comparacion_metodos_emociones_grid.png", dpi=300)
    plt.show()

# ----------------------------------
# Función 1: PCA sobre diferencias neutro - intensidad 04
# ----------------------------------
def compute_emotion_direction_pca(df, emotion_code):
    diffs = []

    grouped = df[df['emotion'] == emotion_code].groupby('person_id')
    for person_id, group in grouped:
        neutral_row = df[(df['person_id'] == person_id) & (df['is_neutral'])]
        emo_row = group[group['intensity'] == '04']

        if len(neutral_row) == 1 and len(emo_row) == 1:
            neutral_vec = neutral_row.iloc[0]['latent_vector']
            emo_vec = emo_row.iloc[0]['latent_vector']
            diff = emo_vec - neutral_vec
            diffs.append(diff)

    if len(diffs) == 0:
        raise ValueError(f"No se encontraron pares neutro-intensidad04 para la emoción {emotion_code}")

    pca = PCA(n_components=1)
    pca.fit(diffs)
    direction = pca.components_[0]

    ## Sanity checks
    ## print(f"[PCA] {emotion_code} → {len(diffs)} vectores")
    return direction

# ----------------------------------
# Función 2: Regresión lineal sobre diferencias neutro - emociones 01 a 04
# ----------------------------------
def compute_emotion_direction_regression(df, emotion_code):
    X = []
    y = []

    grouped = df[df['emotion'] == emotion_code].groupby('person_id')
    for person_id, group in grouped:
        neutral_row = df[(df['person_id'] == person_id) & (df['is_neutral'])]
        if len(neutral_row) != 1:
            continue
        neutral_vec = neutral_row.iloc[0]['latent_vector']

        for _, row in group.iterrows():
            if row['intensity'] in {'01', '02', '03', '04'}:
                emo_vec = row['latent_vector']
                diff = emo_vec - neutral_vec
                X.append(diff)
                y.append(int(row['intensity']))

    if len(X) == 0:
        raise ValueError(f"No se encontraron datos suficientes para la emoción {emotion_code}")

    X = np.array(X)
    y = np.array(y)
    X = X - np.mean(X, axis=0)  # Centrado manual

    lr = LinearRegression()
    lr.fit(X, y)
    direction = lr.coef_

    ## Sanity checks
    ##print(f"[LR] {emotion_code} → {len(X)} vectores")
    ##print(f"LR coef shape for {emotion_code}: {lr.coef_.shape}")
    ##print(f"First 5 values: {lr.coef_[:5]}")
    return direction

def compare_directions_cosine(dir1, dir2):
        v1 = np.array(dir1).reshape(1, -1)
        v2 = np.array(dir2).reshape(1, -1)
        sim = cosine_similarity(v1, v2)[0, 0]
        return sim

def main(align=False, process=False, generate=False, diverse_test=False, celebA=False, verbose=True):
    optional_print("_____________[INICIANDO EJECUCIÓN]_____________", verbose)

    gc.collect()
    torch.cuda.empty_cache()

    # Define the base directory
    base_dir = "/mnt/discoAmpliado/viky/BU_3DFE"

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
        process_all_images(1000, verbose) ## Vamos a hacer de cuenta que esto está bien porque no sé cómo arreglarlo
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
            #print(f"Vector para la emoción {emotion} obtenido con PCA: {directions_pca[emotion]}")
            #print(f"Vector para la emoción {emotion} obtenido con RL: {directions_regression[emotion]}")


    # Mostrar las similaridades
    # print("\nSimilaridades coseno entre direcciones PCA y Regresión:")
    # for emotion, sim in similarities.items():
    #     print(f"{emotion}: {sim:.4f}")

    # ----------------------------------
    # Gráfico para visualizarlas
    # ----------------------------------
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(similarities.keys()), y=list(similarities.values()))
    plt.title("Similaridad coseno entre PCA y Regresión para cada emoción")
    plt.ylabel("Similaridad Coseno")
    plt.xlabel("Emoción")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig("datos/tmp_comparacion_metodos.png", dpi=300)
    plt.show()
    
    #plot_emotion_directions_grid(
    #metadatos_df,
    #directions_pca,
    #directions_regression,
    #similarities,
    #emotion_codes
    #)

    # generate_testing_images(metadatos_df, directions_pca, directions_regression)

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
        print(celeba_sample_df)
        

    optional_print("_____________[EJECUCIÓN FINALIZADA]_____________", verbose)

def generate_diverse_testing_subset(metadatos_df, verbose=True):
    personas = metadatos_df[['person_id', 'gender', 'race']].drop_duplicates()
    distrib = personas.groupby(['gender', 'race']).size().reset_index(name='count')
    
    if(verbose==True):
        # Análisis del dataset
        # Más en general
        print("Nros\n", metadatos_df['race'].value_counts()/25)
        print("Porcentajes\n", metadatos_df['race'].value_counts(normalize=True) * 100)
        print(metadatos_df['gender'].value_counts(normalize=True) * 100)

        # Impreso más lindo
        print(distrib)

    # Creo conjunto de prueba
    prueba_personas = (
        personas.groupby(['gender', 'race'])
        .sample(n=2, random_state=42)  # Cambiá n=1 si querés una versión más chica
    )

    subset_df = metadatos_df[metadatos_df['person_id'].isin(prueba_personas['person_id']) & metadatos_df["is_neutral"] == True]
    return subset_df

def generate_testing_images(metadatos_df, directions_pca, directions_regression):
    output_npz_dir = "images/processed_images"
    os.makedirs(output_npz_dir, exist_ok=True)

    # Emociones y multiplicadores a testear
    emotion_codes = ['HA', 'AN', 'DI', 'FE', 'SA', 'SU']
    multiplicadores_pca = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    multiplicadores_lr = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 10]

    # Imagen base neutra
    neutrals = metadatos_df[metadatos_df["is_neutral"] == True]
    sample_row = neutrals.iloc[2]  ## Acá se puede cambiar la imagen neutra que se usa de base
    base_w = sample_row["latent_vector"] ## Acá NO HAY QUE NORMALIZAR porque manda la imagen al centro del espacio.
    print(f"Shape neutral vector: {base_w.shape}")
    print(f"GENERATING FROM PERSON {sample_row['person_id']}")

    for emotion in emotion_codes:
        delta_pca = directions_pca[emotion]
        delta_lr = directions_regression[emotion]
        print(f"Shape PCA vector: {delta_pca.shape}, shape LR vector {delta_lr.shape}")


        # PCA
        for mult in multiplicadores_pca:
            scaled_delta = mult * delta_pca
            pca_outfile = f"{output_npz_dir}/AA_{emotion}_PCA_x{mult}.npz"
            save_modified_npz(base_w, scaled_delta, pca_outfile)
            generate_one_image_from_npz(pca_outfile, f"AA_{emotion}_PCA_x{mult}")

        # LR
        for mult in multiplicadores_lr:
            scaled_delta = mult * delta_lr
            lr_outfile = f"{output_npz_dir}/AA_{emotion}_LR_x{mult}.npz"
            save_modified_npz(base_w, scaled_delta, lr_outfile)
            generate_one_image_from_npz(lr_outfile, f"AA_{emotion}_LR_x{mult}")

def generate_modified_emotion_images(
    subset_df,
    directions_dict,
    emotion_multipliers,
    method_name,
    output_npz_dir="images/processed_images"
):
    subset_df = subset_df.copy()

    for _, row in subset_df.iterrows():
        person_id = row['person_id']
        neutral_vec = row['latent_vector']

        for emotion, direction in directions_dict.items():
            multiplier = emotion_multipliers.get(emotion, 1.0)
            modified_vec = neutral_vec + multiplier * direction

            # Construir nombre base y path
            base_name = f"AA_prueba_{method_name}_{emotion}_{person_id}"
            npz_path = os.path.join(output_npz_dir, f"{base_name}.npz")

            # Guardar el vector modificado como npz en formato (1, 18, 512)
            modified_w = np.tile(modified_vec[np.newaxis, :], (18, 1))[np.newaxis, :, :]
            np.savez(npz_path, w=modified_w)
            print(npz_path)

            # Generar imagen usando función modular
            generate_one_image_from_npz(npz_path=npz_path, image_name=base_name)

main(False, False, False, False, True, True)

