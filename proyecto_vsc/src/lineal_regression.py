import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------
# Regresión lineal sobre diferencias neutro - emociones 01 a 04
# ----------------------------------
def compute_emotion_direction_regression(df, emotion_code):
    """Calcula el vector de dirección latente de una emoción usando regresión lineal.

    La función opera en el espacio latente (`latent_vector`) y busca la dirección 
    óptima que correlacione la diferencia entre el vector de una imagen emocional 
    y su correspondiente vector neutro con la intensidad numérica de la emoción.

    Proceso:
    1. Filtra las imágenes emocionales para un `emotion_code` dado.
    2. Agrupa por `person_id`.
    3. Para cada persona, calcula el vector de diferencia: $V_{emocional} - V_{neutro}$.
    4. Usa las diferencias vectoriales como características ($X$) y la intensidad 
       numérica (1 a 4) como variable objetivo ($y$).
    5. Ajusta un modelo de regresión lineal ($\mathbf{y} = \mathbf{X} \cdot \mathbf{w}$) 
       para encontrar el vector de pesos $\mathbf{w}$ que mejor predice la intensidad.

    Args:
        df (pandas.DataFrame): DataFrame que contiene los metadatos de las imágenes, 
            incluyendo las columnas 'emotion', 'person_id', 'is_neutral', 'intensity' 
            y 'latent_vector' (el vector 'w' proyectado, shape (512,)).
        emotion_code (str): Código de dos letras de la emoción a analizar (ej. 'HA', 'AN', 'DI').

    Returns:
        numpy.ndarray: El vector de dirección latente (`lr.coef_`) de la emoción, 
            que representa el cambio en el espacio latente asociado con un aumento 
            de una unidad en la intensidad de la emoción (shape (512,)).

    Raises:
        ValueError: Si no se encuentran suficientes datos (pares neutro-emocional con 
            intensidades 01 a 04) para la emoción especificada.
    """

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