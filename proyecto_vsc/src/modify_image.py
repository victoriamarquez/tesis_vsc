import numpy as np
import pandas as pd
import logging
import sys
from pathlib import Path

from image_processing import align_single_image, generate_one_image_from_npz, project_single_image


def modify_image(args):
    input_dir = Path(args.input_folder)
    if not input_dir.exists():
        logging.error("❌ La carpeta no existe.")
        sys.exit(1)

        # Buscar imágenes
    image_files = [
            *input_dir.glob("*.jpg"),
            *input_dir.glob("*.jpeg"),
            *input_dir.glob("*.png"),
        ]

    if not image_files:
        logging.error("❌ No se encontraron imágenes en la carpeta.")
        sys.exit(1)

        # Cargar direcciones
    directions_df = pd.read_csv("/home/vicky/Documents/tesis_vsc/datos/directions_regression.csv")

        # Generar para todas las emociones
    DEFAULT_INTENSITIES = {"HA": 5, "AN": 2, "DI": 0.5, "FE": 2, "SA": 4, "SU": 4.5}

    multiple_emotions_mode = (args.emotion is None and args.intensity is None)

        # Alinear la imagen
    for img_path in image_files:
        logging.info(f"[modify_image] [→] Procesando imagen: {img_path.name}")

        aligned_path = img_path.with_name(img_path.stem + "_01.png")
        w_path = img_path.with_name(img_path.stem + "_projected_w.npz")

        align_single_image(str(img_path))
        project_single_image(img_path)
        
        # Cargar W
        w = np.load(w_path)["w"]

        if multiple_emotions_mode:
            emotions = DEFAULT_INTENSITIES.keys()
        else:
            emotions = [args.emotion]

        for emo in emotions:
            intensity = DEFAULT_INTENSITIES[emo] if multiple_emotions_mode else args.intensity

            logging.debug(f"[modify_image] Usando emoción {emo} con intensidad {intensity}.")

            # Dirección
            direction = directions_df[emo].values.astype(np.float32)
            direction_18 = np.tile(direction, (18, 1)).reshape(1, 18, 512)

            w_mod = w + intensity * direction_18

            out_w_path = img_path.with_name(img_path.stem + f"_mod_{emo}.npz")
            out_img_path = img_path.with_name(img_path.stem + f"_mod_{emo}.png")

            logging.debug(f"[modify_image] Out path npz {out_w_path} out path img {out_img_path}.")

            np.savez(str(out_w_path), w=w_mod)
            generate_one_image_from_npz(out_w_path, out_img_path)

            logging.info(f"[modify_image] [✔] Generada imagen modificada ({emo}) → {out_img_path.name}")