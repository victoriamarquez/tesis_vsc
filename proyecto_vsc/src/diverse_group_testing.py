import os
import numpy as np
from image_processing import generate_one_image_from_npz


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
            generate_one_image_from_npz(npz_path=npz_path, image_name=base_name, outdir_path="/scratch/images/generated_prueba_diversidad")
