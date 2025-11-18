import argparse
import sys

from calculate_vectors import calculate_vectors
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
        action='store_true',
        default=False,
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
        help='Calcula los vectores correspondientes a cada emoción y los almacena en el archivo [COMPLETAR].'
        #TODO: Completar el archivo en el que lo almaceno
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

    # --- Modo 3: test ---
    parser_test = subparsers.add_parser(
        'test',
        help='Ejecuta pruebas del sistema.'
    )
    # Este modo no necesita argumentos específicos adicionales.

    # 3. Análisis de Argumentos y Uso
    args = parser.parse_args()

    print('--- Argumentos Globales ---')
    print(f'Modo seleccionado: {args.mode}')
    print(f'Logging activo: {args.logging}')
    print(f'Verbose: {args.verbose}')
    print('---------------------------')

    # Lógica para cada modo (usando el atributo 'mode' de args)
    if args.mode == 'calculate_vectors':
        print("🛠️ Ejecutando el cálculo de vectores...")
        calculate_vectors(align=True, process=True, generate=True, verbose=args.verbose)
        with open("log.txt", "w") as file:
            file.write("Terminó ejecución calculate_vectors\n")
        pass
    
    elif args.mode == 'modify_image':
        print(f"🖼️ Modificando imagen en la carpeta: **{args.input_folder}**")
        # El parámetro 'input_folder' solo está disponible cuando 'mode' es 'modify_image'
        pass
        
    elif args.mode == 'test':
        print("✅ Ejecutando pruebas...")
        execute_tests()
        pass

if __name__ == '__main__':
    # Esto permite que el script se ejecute directamente
    # Si quisieras simular la ejecución de la línea de comandos, usa:
    # args = parser.parse_args(['modify_image', '/home/user/images', '--no-logging'])
    main()
