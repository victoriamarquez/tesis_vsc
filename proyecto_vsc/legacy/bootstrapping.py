from helpers import *
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt
from statistics import mean

def umbralizar_matriz(matriz_similitud, umbral=0.15):
    # Convertir la matriz de similitud a un numpy array si no lo es
    matriz_similitud = np.array(matriz_similitud)
    
    # Aplicar el umbral: cualquier valor mayor o igual al umbral se convierte en 1, los demás en 0.
    matriz_umbralizada = np.where(matriz_similitud >= umbral, 1, 0)
    
    # Calcular el promedio de los valores umbralizados
    promedio_umbralizado = np.mean(matriz_umbralizada)
    
    return promedio_umbralizado, matriz_umbralizada

# Función para calcular la matriz de similitud coseno
def calcular_matriz_similitud(vectores):
    return 1-cosine_similarity(vectores)

# Función para extraer vectores de una columna y devolverlos como una lista
def extraer_vectores_emocion(df, emocion):
    return np.array([vector for vector in df[emocion]])

# Función para extraer un conjunto aleatorio de vectores
def extraer_vectores_aleatorios(df, cantidad):
    # Tomar una muestra aleatoria de 'cantidad' vectores de cualquier columna (emociones) del DataFrame
    columnas = df.columns
    vectores_aleatorios = []
    
    for _ in range(cantidad):
        emocion_random = random.choice(columnas)  # Elegir una emoción aleatoria
        id_random = random.choice(df.index)  # Elegir una persona aleatoria
        vectores_aleatorios.append(df.loc[id_random, emocion_random])  # Agregar el vector aleatorio
    
    return np.array(vectores_aleatorios)

# Función principal

def evaluar_similitud(df, emocion, n_iteraciones=100, umbral=0.15, graficar=False):
    # 1. Extraer vectores de la emoción seleccionada (ej: "FE" para felicidad)
    vectores_emocion = extraer_vectores_emocion(df, emocion)
    
    # 2. Calcular la matriz de similitud coseno entre estos vectores
    matriz_similitud_emocion = calcular_matriz_similitud(vectores_emocion)
   
    # 3. Aplicar la función de umbralización que ya tienes
    promedio_umbralizado_emocion, _ = umbralizar_matriz(matriz_similitud_emocion, umbral)
    
    # Lista para almacenar los promedios umbralizados de conjuntos aleatorios
    promedios_aleatorios = []
    
    # 4. Repetir el cálculo para 'n_iteraciones' conjuntos aleatorios
    for _ in range(n_iteraciones):
        # Extraer un conjunto aleatorio de vectores (del mismo tamaño que los vectores de emoción)
        vectores_aleatorios = extraer_vectores_aleatorios(df, len(vectores_emocion))
        
        # Calcular la matriz de similitud coseno para el conjunto aleatorio
        matriz_similitud_aleatoria = calcular_matriz_similitud(vectores_aleatorios)
        
        # Aplicar la función de umbralización a la matriz aleatoria
        promedio_umbralizado_aleatorio, _ = umbralizar_matriz(matriz_similitud_aleatoria, umbral)
        
        # Guardar el resultado
        promedios_aleatorios.append(promedio_umbralizado_aleatorio)
    
    # 5. Graficar los resultados
    if(graficar):
        plt.figure(figsize=(10, 6))
        plt.hist(promedios_aleatorios, bins=30, alpha=0.7, label='Promedios Aleatorios')
        plt.axvline(promedio_umbralizado_emocion, color='r', linestyle='dashed', linewidth=2, label=f'Promedio {emocion}')
        #plt.axvline(0, color='g', linestyle='dashed', linewidth=2, label=f'0')
        plt.axvline(mean(promedios_aleatorios), color='b', linestyle='dashed', linewidth=2, label=f'Media de los promedios aleatorios')
        plt.suptitle(f'Comparación entre promedios aleatorios y el promedio de {emocion}')
        plt.title(f'Usando {n_iteraciones} iteraciones y umbral de {umbral}.', fontsize=10)
        plt.xlabel('Promedio Umbralizado')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.show()
    
    return promedio_umbralizado_emocion, promedios_aleatorios

def bootstrapping_todas_emociones(emociones_total_PCA_normalizado, emociones_total_LR_normalizado, emociones, iteraciones=1000):

    for emocion in emociones:

        promedio_emocion, promedios_aleatorios = evaluar_similitud(emociones_total_PCA_normalizado, emocion, iteraciones)

        optional_print(f'Promedio umbralizado para {emocion} (PCA): {promedio_emocion}')
        optional_print(f"Promedio de los promedios aleatorios (PCA): {mean(promedios_aleatorios)}")

        promedio_emocion, promedios_aleatorios = evaluar_similitud(emociones_total_LR_normalizado, emocion, iteraciones)

        optional_print(f'Promedio umbralizado para {emocion} (LR): {promedio_emocion}')
        optional_print(f"Promedio de los promedios aleatorios (LR): {mean(promedios_aleatorios)}")

def umbrales_optimos(df, iteraciones=100):
    # Lista de emociones a evaluar
    emociones = df.keys().unique().tolist()
    
    # Rango de umbrales
    umbrales = np.arange(0, 1.01, 0.01)
    
    # Crear un dataframe para almacenar los resultados
    resultados = pd.DataFrame(columns=["Emocion", "Umbral", "Media_Aleatorios", "Promedio_Emocion", "Diferencia"])
    
    # Iterar sobre cada emoción
    for emocion in emociones:
        for umbral in umbrales:
            # Ejecutar la función evaluar_similitud
            output_emocion, outputs_aleatorios = evaluar_similitud(df, emocion, iteraciones, umbral)
            
            # Calcular la media de outputs_aleatorios
            media_aleatorios = np.mean(outputs_aleatorios)
            
            # Calcular la diferencia
            diferencia = abs(output_emocion - media_aleatorios)
            
            # Almacenar los resultados en el dataframe
            resultados = resultados.append({
                "Emocion": emocion,
                "Umbral": umbral,
                "Media_Aleatorios": media_aleatorios,
                "Promedio_Emocion": output_emocion,
                "Diferencia": diferencia
            }, ignore_index=True)
    
    # Encontrar los umbrales que maximizan la diferencia para cada emoción
    umbrales_optimos = resultados.loc[resultados.groupby("Emocion")["Diferencia"].idxmax()]
    # Mostrar los umbrales óptimos
    optional_print(umbrales_optimos)
    return resultados