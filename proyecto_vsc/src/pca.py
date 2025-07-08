from sklearn.decomposition import PCA

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
