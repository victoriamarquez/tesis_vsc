import logging
import os
import numpy as np
from image_processing import generate_one_image_from_npz


def generate_diverse_testing_subset(metadatos_df):
    """Crea un subconjunto de imágenes neutras equilibrado por género y raza para pruebas.

    Esta función selecciona aleatoriamente un número fijo de personas (`n=2` por defecto) 
    para cada combinación única de género y raza (`'gender', 'race'`) presente en 
    el DataFrame de metadatos. Luego, filtra solo las imágenes con expresión neutra 
    (`"is_neutral" == True`) de esas personas seleccionadas para conformar el conjunto de prueba.

    Args:
        metadatos_df (pandas.DataFrame): DataFrame que contiene los metadatos de las 
            imágenes, incluyendo las columnas `person_id`, `gender`, `race` e `is_neutral`.

    Returns:
        pandas.DataFrame: Un DataFrame que es un subconjunto de `metadatos_df`, 
            conteniendo solo las filas de imágenes neutras pertenecientes al grupo 
            de personas seleccionadas aleatoriamente por género/raza.
    """

    personas = metadatos_df[['person_id', 'gender', 'race']].drop_duplicates()
    distrib = personas.groupby(['gender', 'race']).size().reset_index(name='count')
    

    # Análisis del dataset
    # Más en general
    logging.debug(f"[Diverse] Cantidad de personas pertenecientes a cada raza: \n{metadatos_df['race'].value_counts()/25}")
    logging.debug(f"[Diverse] Porcentajes: \n Race: \n{metadatos_df['race'].value_counts(normalize=True) * 100}\n Gender: \n{metadatos_df['gender'].value_counts(normalize=True) * 100}")

    # Impreso más lindo
    logging.debug(f"[Diverse] Distribución de género y raza en números: \n{distrib}")

    # Creo conjunto de prueba
    prueba_personas = (
        personas.groupby(['gender', 'race'])
        .sample(n=2, random_state=42)  # Cambiar n para modificar cant de personas
    )

    subset_df = metadatos_df[metadatos_df['person_id'].isin(prueba_personas['person_id']) & metadatos_df["is_neutral"] == True]
    return subset_df

def generate_modified_emotion_images(
    subset_df,
    directions_dict,
    emotion_multipliers,
    method_name,
    output_npz_dir="images/processed_images"
):
    """Aplica vectores de dirección emocional a vectores latentes neutros y genera las imágenes.

    Esta función toma un DataFrame de prueba (generalmente un subconjunto diverso de 
    imágenes neutras), itera sobre cada imagen, aplica todos los vectores de dirección 
    emocional provistos, guarda el nuevo vector latente modificado como NPZ, y luego 
    sintetiza la imagen PNG resultante utilizando la función de generación modular.

    Args:
        subset_df (pandas.DataFrame): DataFrame con el subconjunto de imágenes de prueba. 
            Debe contener las columnas 'person_id' y 'latent_vector' (el vector 'w' neutro).
        directions_dict (dict): Diccionario que mapea la emoción (str) al vector 
            diferencial de dirección (numpy.ndarray de shape (512,)).
        emotion_multipliers (dict): Diccionario que mapea la emoción (str) al escalar 
            de intensidad (`multiplier`) para la manipulación.
        method_name (str): Nombre del método utilizado para generar las direcciones 
            (ej. 'GANSpace', 'InterFaceGAN'), usado para nombrar los archivos de salida.
        output_npz_dir (str, optional): Directorio donde se guardarán los archivos 
            NPZ con los vectores latentes modificados.

    Returns:
        None: La función no devuelve nada, pero genera archivos .npz y .png en los 
            directorios de salida configurados.
    
    Raises:
        KeyError: Si alguna clave de emoción no está presente en `emotion_multipliers`.
    """
    
    subset_df = subset_df.copy()

    logging.info(f"[Diverse] [→] Generando imágenes modificadas.")

    for _, row in subset_df.iterrows():
        person_id = row['person_id']
        neutral_vec = row['latent_vector']

        for emotion, direction in directions_dict.items():
            multiplier = emotion_multipliers.get(emotion, 1.0)
            modified_vec = neutral_vec + multiplier * direction

            # Construir nombre base y path
            base_name = f"testing_{method_name}_{emotion}_{person_id}"
            npz_path = os.path.join(output_npz_dir, f"{base_name}.npz")

            # Guardar el vector modificado como npz en formato (1, 18, 512)
            modified_w = np.tile(modified_vec[np.newaxis, :], (18, 1))[np.newaxis, :, :]
            np.savez(npz_path, w=modified_w)

            # Generar imagen usando función modular
            generate_one_image_from_npz(npz_path=npz_path, outdir_path="/scratch/images/generated_prueba_diversidad")
    logging.info("[Diverse] [✔] Imagenes modificadas generadas.")
