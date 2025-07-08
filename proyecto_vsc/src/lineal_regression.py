import numpy as np
from sklearn.linear_model import LinearRegression

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