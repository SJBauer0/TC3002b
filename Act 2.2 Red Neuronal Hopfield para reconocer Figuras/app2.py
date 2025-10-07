# ==================================
# Sebastian Jimenez Bauer 
# A01708830
# ==================================
import random

def cargar_patrones(lista_archivos):
    """
    Carga patrones desde archivos de texto y los convierte a formato bipolar (1, -1).
    """
    patrones_cargados = []
    for nombre_archivo in lista_archivos:
        try:
            with open(nombre_archivo, 'r') as f:
                matriz = []
                for linea in f:
                    # Convierte '1' a 1 y cualquier otra cosa (como '0') a -1
                    fila = [1 if int(n) == 1 else -1 for n in linea.split()]
                    matriz.append(fila)
                
                # Aplana la matriz 10x10 a un vector de 100 neuronas
                vector_plano = [pixel for fila in matriz for pixel in fila]
                patrones_cargados.append(vector_plano)
                print(f"Patrón '{nombre_archivo}' cargado ({len(vector_plano)} neuronas).")
        except FileNotFoundError:
            print(f"¡Error! No se encontró el archivo: {nombre_archivo}")
        except Exception as e:
            print(f"Ocurrió un error al leer {nombre_archivo}: {e}")
    return patrones_cargados

def imprimir_patron(vector):
    """
    Imprime un vector de 100 elementos como una matriz visual de 10x10.
    """
    if len(vector) != 100:
        print("El vector no tiene 100 elementos para imprimir como 10x10.")
        return
    for i in range(10):
        # Imprime '■' para 1 y un espacio para -1
        linea = " ".join(['■' if vector[i*10 + j] == 1 else ' ' for j in range(10)])
        print(linea)

archivos_a_memorizar = ["uno_maya.txt", "dos_maya.txt", "tres_maya.txt", "cuatro_maya.txt", "cinco_maya.txt"]
patrones_memoria = cargar_patrones(archivos_a_memorizar)

num_neuronas = len(patrones_memoria[0])

W = [[0 for _ in range(num_neuronas)] for _ in range(num_neuronas)]

for p in patrones_memoria:
    for i in range(num_neuronas):
        for j in range(num_neuronas):
            W[i][j] += p[i] * p[j]

for i in range(num_neuronas):
    W[i][i] = 0

print("\nMatriz de pesos generada para los patrones cargados.")

patron_original = patrones_memoria[0]
patron_ruidoso = list(patron_original)

pixeles_a_cambiar = 15
indices_a_cambiar = random.sample(range(num_neuronas), pixeles_a_cambiar)
for idx in indices_a_cambiar:
    patron_ruidoso[idx] *= -1

print(f"\n--- Patrón original ('uno') ---")
imprimir_patron(patron_original)

print(f"\n--- Patrón de entrada con {pixeles_a_cambiar} píxeles de ruido ---")
imprimir_patron(patron_ruidoso)

print("\nIniciando proceso de reconocimiento...")

s = patron_ruidoso.copy()
max_iterations = 10
estabilizado = False

for i in range(max_iterations):
    s_old = s.copy()

    update_order = list(range(num_neuronas))
    random.shuffle(update_order)
    
    for neuron_idx in update_order:
        activation = 0
        for j in range(num_neuronas):
            activation += W[neuron_idx][j] * s[j]
        
        s[neuron_idx] = 1 if activation >= 0 else -1
            
    if s == s_old:
        print(f"La red se estabilizó en la iteración {i + 1}.")
        estabilizado = True
        break

if not estabilizado:
    print("Se alcanzó el número máximo de iteraciones.")

print(f"\n--- Patrón reconocido final ---")
imprimir_patron(s)

if s == patron_original:
    print("\n¡Éxito!  El patrón recordado coincide con el original.")
else:
    print("\nFallo. El patrón recordado no coincide con el original.")