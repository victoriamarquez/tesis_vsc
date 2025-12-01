import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from helpers import save_modified_npz
from image_processing import generate_one_image_from_npz


def plot_emotion_directions_grid(df, directions_pca, directions_regression, similarities, emotion_codes):
    """
    [NO USADA!?]
    Genera una cuadrícula de gráficos que comparan las direcciones de emoción obtenidas 
    por PCA y Regresión Lineal contra las diferencias reales en el espacio latente.

    Para cada emoción:
    1. Calcula el vector de diferencia (emoción - neutro, con intensidad '04') para cada persona.
    2. Utiliza PCA para reducir los vectores de diferencia a 2D.
    3. Proyecta las direcciones PCA y de Regresión Lineal al mismo espacio 2D.
    4. Grafica las diferencias reales (puntos dispersos) y las direcciones de los métodos (flechas).
    5. Muestra la similitud coseno entre las dos direcciones calculadas en el título.

    Args:
        df (pandas.DataFrame): DataFrame de metadatos con la columna 'latent_vector', 
            incluyendo las columnas 'person_id', 'emotion', 'is_neutral' e 'intensity'.
        directions_pca (dict): Diccionario de vectores de dirección obtenidos por PCA 
            (clave: código de emoción, valor: np.ndarray).
        directions_regression (dict): Diccionario de vectores de dirección obtenidos por 
            Regresión Lineal (clave: código de emoción, valor: np.ndarray).
        similarities (dict): Diccionario que contiene la similitud coseno entre los 
            vectores de PCA y Regresión Lineal para cada emoción.
        emotion_codes (list): Lista de códigos de emoción a graficar (ej. ['HA', 'AN', ...]).

    Returns:
        None: La función muestra el gráfico y lo guarda como un archivo PNG.
    """

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

def compare_directions_cosine(dir1, dir2):
    """Calcula la similitud coseno entre dos vectores de dirección latente.

    Usada en calculate_vectors para el cálculo de similutudes coseno.
    
    La similitud coseno mide el coseno del ángulo entre dos vectores, indicando 
    qué tan similar es la dirección de ambos, ignorando su magnitud. 
    Un valor cercano a 1 indica alta similitud direccional.

    Args:
        dir1 (array-like): Primer vector de dirección (numpy.ndarray o lista).
        dir2 (array-like): Segundo vector de dirección (numpy.ndarray o lista).

    Returns:
        float: El valor de la similitud coseno entre los dos vectores.
    """
    v1 = np.array(dir1).reshape(1, -1)
    v2 = np.array(dir2).reshape(1, -1)
    sim = cosine_similarity(v1, v2)[0, 0]
    return sim

def generate_testing_images_for_multipliers(metadatos_df, directions_pca, directions_regression):
    """
    [LEGACY]
    Genera una serie de imágenes sintéticas aplicando direcciones emocionales con 
    diferentes niveles de intensidad (multiplicadores).

    La función toma una única imagen neutra de referencia (iloc[2]), aplica las 
    direcciones obtenidas por PCA y Regresión Lineal, escalándolas con un rango 
    predeterminado de multiplicadores para cada método. Esto permite visualizar y 
    comparar el efecto de la intensidad y la calidad de la dirección de cada método.

    Args:
        metadatos_df (pandas.DataFrame): DataFrame de metadatos de imágenes (con la 
            columna 'latent_vector' y 'is_neutral') para seleccionar la imagen base neutra.
        directions_pca (dict): Diccionario de vectores de dirección obtenidos por PCA.
        directions_regression (dict): Diccionario de vectores de dirección obtenidos 
            por Regresión Lineal.

    Returns:
        None: La función guarda archivos .npz y genera las imágenes PNG correspondientes 
            en sus directorios de salida.
    """

    output_npz_dir = "images/processed_images"
    os.makedirs(output_npz_dir, exist_ok=True)

    # Emociones y multiplicadores a testear
    emotion_codes = ['HA', 'AN', 'DI', 'FE', 'SA', 'SU']
    multiplicadores_pca = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    multiplicadores_lr = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 10]

    # Imagen base neutra
    neutrals = metadatos_df[metadatos_df["is_neutral"] == True]
    sample_row = neutrals.iloc[2]  # Acá se puede cambiar la imagen neutra que se usa de base
    base_w = sample_row["latent_vector"] # Acá NO HAY QUE NORMALIZAR porque manda la imagen al centro del espacio.
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
