import argparse
import gc
import logging
import torch

from calculate_vectors import calculate_vectors
from modify_image import modify_image
from testing import execute_tests

def main():

    # 1. Configuración del ArgumentParser Principal
    parser = argparse.ArgumentParser(
        description='Tesis de licenciatura: Herramienta de generación de imágenes con diferentes expresiones faciales.',
        epilog='Usa "<subcomando> -h" para más ayuda en un modo específico.'
    )

    # Argumentos Globales (Aplican a todos los modos)
    parser.add_argument(
        '-v', '--verbose',
        action='store_false',
        default=True,
        help='Mostrar mensajes detallados de ejecución.'
    )

    # El 'action="store_false"' invierte el comportamiento,
    # el default es True (logging activo), y al usar --no-logging se establece en False.
    parser.add_argument(
        '--no-logging',
        dest='logging',
        action='store_false',
        default=True,
        help='Desactiva el registro de eventos (logging).'
    )

    # 2. Creación de Subcomandos
    subparsers = parser.add_subparsers(
        dest='mode',
        required=True, # Hace que sea obligatorio seleccionar un modo.
        help='Selecciona el modo de operación.'
    )

    # --- Modo 1: calculate vectors ---
    parser_calculate = subparsers.add_parser(
        'calculate_vectors',
        help='Calcula los vectores correspondientes a cada emoción y los almacena en el archivo datos/directions_regression.csv.'
    )
    # Este modo no necesita argumentos específicos adicionales.

    # --- Modo 2: modify image ---
    parser_modify = subparsers.add_parser(
        'modify_image',
        help='Modifica la imagen en una carpeta específica.'
    )
    # Argumento Posicional Específico para este modo
    parser_modify.add_argument(
        'input_folder',
        type=str,
        help='Ruta a la carpeta que contiene la imagen a modificar.'
    )

    parser_modify.add_argument(
        "--emotion",
        type=str,
        choices=["HA", "AN", "DI", "FE", "SA", "SU"],
        help="Emoción a modificar."
    )

    parser_modify.add_argument(
        "--intensity",
        type=float,
        help="Intensidad del cambio emocional."
    )

    # --- Modo 3: test ---
    parser_test = subparsers.add_parser(
        'test',
        help='Ejecuta pruebas del sistema.'
    )
    # Este modo no necesita argumentos específicos adicionales.

    # 3. Análisis de Argumentos y Uso
    args = parser.parse_args()

    logging.basicConfig( 
        level=logging.INFO if args.logging==True else logging.ERROR,           
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("tesis.log", mode='w'),  # Logs to a file named tesis.log, clears on start
            logging.StreamHandler()         # Logs to the console (terminal)
        ] if args.verbose==True else [logging.FileHandler("tesis.log")]
    )

    logging.info('--- Argumentos Globales ---')
    logging.info(f'Modo seleccionado: {args.mode}')
    logging.info(f'Logging activo: {args.logging}')
    logging.info(f'Verbose: {args.verbose}')
    logging.info('---------------------------')

    gc.collect()
    torch.cuda.empty_cache()

    # Lógica para cada modo (usando el atributo 'mode' de args)
    if args.mode == 'calculate_vectors':
        logging.info("🛠️ Ejecutando el cálculo de vectores...")
        calculate_vectors(align=False, process=True, generate=False)
        logging.info("🛠️ Finalizó ejecución calculate_vectors")
        pass
    
    elif args.mode == 'modify_image':
        # El parámetro 'input_folder' solo está disponible cuando 'mode' es 'modify_image'
        logging.info(f"🖼️ Modificando imagen en la carpeta: {args.input_folder}")
        modify_image(args)
        logging.info(f"🖼️ Finalizó modificación de imagen en la carpeta: {args.input_folder}")
        pass
        
    elif args.mode == 'test':
        logging.info("✅ Ejecutando pruebas...")
        execute_tests()
        logging.info("✅ Finalizó ejecución de pruebas.")
        pass

if __name__ == '__main__':
    main()
