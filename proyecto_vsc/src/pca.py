from sklearn.decomposition import PCA

# ----------------------------------
# PCA sobre diferencias neutro - intensidad 04
# ----------------------------------
def compute_emotion_direction_pca(df, emotion_code):
    """
    Calcula el vector de dirección latente principal de una emoción utilizando PCA.

    La función identifica el vector que explica la mayor varianza en las diferencias 
    de vectores latentes entre las imágenes con máxima intensidad de la emoción 
    especificada y sus correspondientes imágenes neutras.

    Proceso:
    1. Filtra las imágenes emocionales con máxima intensidad ('04') para un 
       `emotion_code` dado.
    2. Para cada `person_id`, calcula el vector de diferencia: $V_{emocional, I04} - V_{neutro}$.
    3. Aplica el **Análisis de Componentes Principales (PCA)** sobre el conjunto de 
       vectores de diferencia.
    4. El vector de dirección se define como el **primer componente principal** ($n\_components=1$), que representa la dirección de máxima variabilidad.

    Args:
        df (pandas.DataFrame): DataFrame que contiene los metadatos de las imágenes, 
            incluyendo las columnas 'emotion', 'person_id', 'is_neutral', 'intensity' 
            y 'latent_vector' (el vector 'w' proyectado, shape (512,)).
        emotion_code (str): Código de dos letras de la emoción a analizar (ej. 'HA', 'AN').

    Returns:
        numpy.ndarray: El primer componente principal (`pca.components_[0]`), que 
            es el vector de dirección latente de la emoción (shape (512,)).

    Raises:
        ValueError: Si no se encuentran pares de imágenes (neutro - emoción intensidad 04) 
            suficientes para la emoción especificada.
    """

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
