import random

def cargar_patrones(lista_archivos):
    patrones_cargados = []
    for nombre_archivo in lista_archivos:
        try:
            with open(nombre_archivo, 'r') as f:
                matriz = []
                for linea in f:
                    fila = [int(n) for n in linea.split()]
                    fila = [1 if n == 1 else -1 for n in fila]
                    matriz.append(fila)
                
                vector_plano = [pixel for fila in matriz for pixel in fila]
                patrones_cargados.append(vector_plano)
                print(f"Patrón '{nombre_archivo}' cargado ({len(vector_plano)} neuronas).")
        except FileNotFoundError:
            print(f"¡Error! No se encontró el archivo: {nombre_archivo}")
        except Exception as e:
            print(f"Ocurrió un error al leer {nombre_archivo}: {e}")
    return patrones_cargados

def entrenar_hopfield(patrones):
    if not patrones:
        print("La lista de patrones está vacía. No se puede entrenar.")
        return []
    num_neuronas = len(patrones[0])
    W = [[0 for _ in range(num_neuronas)] for _ in range(num_neuronas)]
    for p in patrones:
        for i in range(num_neuronas):
            for j in range(num_neuronas):
                if i != j:
                    W[i][j] += (p[i] * p[j]) / num_neuronas 
    print("Entrenamiento completado. Matriz de pesos generada.")
    return W

def funcion_activacion(valor):
    return 1 if valor >= 0 else -1

def distancia_hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def imprimir_patron(vector, ancho=10):
    for i in range(ancho):
        linea = " ".join(['■' if vector[i*ancho + j] == 1 else ' ' for j in range(ancho)])
        print(linea)

def reconocer_patron(W, patron_entrada, max_iteraciones=100):
    num_neuronas = len(patron_entrada)
    patron_actual = list(patron_entrada)
    for _ in range(max_iteraciones):
        patron_anterior = list(patron_actual)
        for i in range(num_neuronas):
            suma_ponderada = sum(W[i][j] * patron_anterior[j] for j in range(num_neuronas))
            patron_actual[i] = funcion_activacion(suma_ponderada)
        if patron_actual == patron_anterior:
            return patron_actual
    return patron_actual

def main():
    archivos_a_memorizar = ["uno_maya.txt", "dos_maya.txt", "tres_maya.txt", "cuatro_maya.txt", "cinco_maya.txt"]
    patrones_memoria = cargar_patrones(archivos_a_memorizar)

    if not patrones_memoria:
        print("No se cargaron patrones. Saliendo.")
        return

    matriz_pesos = entrenar_hopfield(patrones_memoria)

    print("\n======= PRUEBA DE RECONOCIMIENTO =======\n")

    exitos = 0

    for idx, patron_original in enumerate(patrones_memoria):
        nombre = archivos_a_memorizar[idx]
        patron_ruidoso = list(patron_original)

        # Añadir ruido: cambiar 2 bits aleatorios
        indices_a_cambiar = random.sample(range(len(patron_original)), 18)
        for i in indices_a_cambiar:
            patron_ruidoso[i] *= -1

        print(f"\n--- {nombre.upper()} ---")
        print("Patrón con ruido (entrada):")
        imprimir_patron(patron_ruidoso)
        
        patron_recordado = reconocer_patron(matriz_pesos, patron_ruidoso)
        
        print("\nPatrón recordado (salida):")
        imprimir_patron(patron_recordado)

        d = distancia_hamming(patron_original, patron_recordado)
        print(f"\nDistancia de Hamming: {d}")

        if d == 0:
            print("El patrón recordado coincide con el original.")
            exitos += 1
        else:
            print("El patrón recordado difiere del original.")

    print("\n======= RESULTADOS FINALES =======")
    print(f"Patrones recordados correctamente: {exitos}/{len(patrones_memoria)}")

if __name__ == "__main__":
    main()
