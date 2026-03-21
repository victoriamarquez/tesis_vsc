"""
Script para pre-filtrar imágenes de CelebA candidatas para revisión manual.

Lee el archivo de atributos de CelebA, descarta imágenes con características
no deseadas (borrosas, con sombrero, con anteojos, sonriendo), toma una muestra
aleatoria, copia las imágenes a una carpeta de salida y genera una galería HTML
para facilitar la revisión visual en la máquina local.

Uso típico (en la compu remota):
    python filter_celeba_candidates.py \\
        --attr-file /home/vicky/Documents/tesis_vsc/list_attr_celeba.txt \\
        --images-dir /home/vicky/Documents/tesis_vsc/images/CelebA/img_align_celeba \\
        --output-dir /home/vicky/Documents/tesis_vsc/images/CelebA/candidatas_revision \\
        --n 20

Luego copiar la carpeta de salida a tu máquina local para revisión:
    scp -r vicky@<ip>:/home/vicky/Documents/tesis_vsc/images/CelebA/candidatas_revision ./
    (Abrir candidatas_revision/galeria.html en el navegador)
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

# Atributos que indican mala calidad u oclusión. Imágenes con cualquiera de
# estos atributos presentes (valor=1) serán descartadas.
# "Smiling" se excluye porque queremos expresión neutral como punto de partida
# para aplicar las emociones, igual que en el dataset BU-3DFE.
ATTRIBUTES_TO_EXCLUDE = ["Blurry", "Wearing_Hat", "Eyeglasses", "Smiling"]

HTML_GALLERY_TEMPLATE = """\
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>CelebA — Candidatas para revisión</title>
  <style>
    body {{ font-family: sans-serif; background: #1a1a1a; color: #eee; padding: 20px; }}
    h1 {{ margin-bottom: 4px; }}
    p.subtitle {{ color: #aaa; margin-top: 0; margin-bottom: 24px; font-size: 14px; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .card {{
      background: #2a2a2a; border-radius: 6px; padding: 8px;
      width: 160px; text-align: center;
    }}
    .card img {{
      width: 144px; height: 144px; object-fit: cover;
      border-radius: 4px; display: block;
    }}
    .card span {{
      font-size: 11px; color: #aaa; word-break: break-all;
      margin-top: 6px; display: block;
    }}
  </style>
</head>
<body>
  <h1>CelebA — Candidatas para revisión manual</h1>
  <p class="subtitle">
    {count} imágenes · filtros aplicados: {filters} · seed {seed}
  </p>
  <div class="grid">
{cards}
  </div>
</body>
</html>
"""

HTML_CARD_TEMPLATE = """\
    <div class="card">
      <img src="{filename}" alt="{filename}">
      <span>{filename}</span>
    </div>"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pre-filtra imágenes de CelebA para revisión manual."
    )
    parser.add_argument(
        "--attr-file",
        required=True,
        help="Ruta al archivo list_attr_celeba.txt"
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Ruta a la carpeta con las imágenes de CelebA (img_align_celeba)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Carpeta donde se copiarán las imágenes seleccionadas y la galería HTML"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Cantidad de imágenes a seleccionar (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad del muestreo aleatorio (default: 42)"
    )
    return parser.parse_args()


def load_celeba_attributes(attr_path):
    """Carga el archivo de atributos de CelebA en un DataFrame."""
    with open(attr_path, "r") as attribute_file:
        lines = attribute_file.readlines()
    attribute_names = lines[1].strip().split()
    rows = []
    for line in lines[2:]:
        parts = line.strip().split()
        filename = parts[0]
        attributes = list(map(int, parts[1:]))
        rows.append([filename] + attributes)
    return pd.DataFrame(rows, columns=["file_name"] + attribute_names)


def filter_candidates(dataframe):
    """Descarta imágenes con atributos no deseados para el experimento."""
    inclusion_mask = pd.Series(True, index=dataframe.index)
    for attribute in ATTRIBUTES_TO_EXCLUDE:
        if attribute in dataframe.columns:
            inclusion_mask &= (dataframe[attribute] != 1)
    return dataframe[inclusion_mask].reset_index(drop=True)


def sample_candidates(dataframe, count, seed):
    """Toma una muestra aleatoria reproducible del conjunto filtrado."""
    sample_size = min(count, len(dataframe))
    return dataframe.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def copy_selected_images(dataframe, images_directory, output_directory):
    """Copia las imágenes seleccionadas a la carpeta de salida."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    missing_files = []
    for file_name in dataframe["file_name"]:
        source_path = Path(images_directory) / file_name
        if source_path.exists():
            shutil.copy(source_path, output_directory / file_name)
            copied_count += 1
        else:
            missing_files.append(file_name)
    if missing_files:
        print(f"  ⚠️  No se encontraron {len(missing_files)} imágenes en {images_directory}")
        for missing_file in missing_files[:5]:
            print(f"     - {missing_file}")
        if len(missing_files) > 5:
            print(f"     ... y {len(missing_files) - 5} más.")
    return copied_count


def generate_html_gallery(dataframe, output_directory, seed):
    """Genera una galería HTML estática para revisión visual de las imágenes."""
    cards = "\n".join(
        HTML_CARD_TEMPLATE.format(filename=file_name)
        for file_name in dataframe["file_name"]
    )
    html_content = HTML_GALLERY_TEMPLATE.format(
        count=len(dataframe),
        filters=", ".join(ATTRIBUTES_TO_EXCLUDE),
        seed=seed,
        cards=cards,
    )
    gallery_path = Path(output_directory) / "galeria.html"
    gallery_path.write_text(html_content, encoding="utf-8")
    return gallery_path


def save_selected_filenames(dataframe, output_directory):
    """Guarda la lista de nombres de archivo seleccionados en un .txt."""
    list_path = Path(output_directory) / "candidatas.txt"
    dataframe["file_name"].to_csv(list_path, index=False, header=False)
    return list_path


def main():
    args = parse_arguments()

    print(f"[1/4] Cargando atributos desde {args.attr_file} ...")
    attributes_dataframe = load_celeba_attributes(args.attr_file)
    print(f"      Total de imágenes en el dataset: {len(attributes_dataframe):,}")

    print(f"[2/4] Filtrando por: {', '.join(ATTRIBUTES_TO_EXCLUDE)}")
    filtered_dataframe = filter_candidates(attributes_dataframe)
    print(f"      Imágenes luego del filtrado: {len(filtered_dataframe):,}")

    print(f"[3/4] Muestreando {args.n} imágenes (seed={args.seed}) ...")
    sampled_dataframe = sample_candidates(filtered_dataframe, args.n, args.seed)

    print(f"[4/4] Copiando imágenes y generando galería en {args.output_dir} ...")
    copied_count = copy_selected_images(sampled_dataframe, args.images_dir, args.output_dir)
    gallery_path = generate_html_gallery(sampled_dataframe, args.output_dir, args.seed)
    list_path = save_selected_filenames(sampled_dataframe, args.output_dir)

    print(f"\n✅ Listo.")
    print(f"   {copied_count} imágenes copiadas en:  {args.output_dir}")
    print(f"   Galería HTML generada en:           {gallery_path}")
    print(f"   Lista de candidatas guardada en:    {list_path}")
    print(f"\n   Para copiar la carpeta a tu máquina local y revisar:")
    print(f"   scp -r vicky@<ip>:{args.output_dir} .")
    print(f"   Luego abrir galeria.html en el navegador.")


if __name__ == "__main__":
    main()
