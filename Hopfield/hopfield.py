import random

x1 = [1, 1, 1, -1]
x2 = [-1, -1, -1, 1]

x1t_x1 = []
for i in x1:
    new_row = []
    for j in x1:      
        new_row.append(i * j)
    x1t_x1.append(new_row)

x2t_x2 = []
for i in x2:
    new_row = []
    for j in x2:      
        new_row.append(i * j)
    x2t_x2.append(new_row)

sum_matrix = []
for i in range(len(x1t_x1)):
    new_row = []
    for j in range(len(x1t_x1[0])):
        new_row.append(x1t_x1[i][j] + x2t_x2[i][j])
    sum_matrix.append(new_row)

for i in range(len(sum_matrix)):
    sum_matrix[i][i] = 0

print("Matriz de pesos:")
for row in sum_matrix:
    print(row)

input_str = input("\nIngresa el patrón de prueba (ej: 1 -1 1 -1): ").split()
test_pattern = [int(i) for i in input_str]

print(f"\nPatrón de entrada a reconocer:\n{test_pattern}\n")

s = test_pattern.copy()
num_neurons = len(s)
max_iterations = 10

for i in range(max_iterations):
    print(f"Iteración {i + 1}: {s}")
    s_old = s.copy()

    update_order = list(range(num_neurons))
    random.shuffle(update_order)
    
    for neuron_idx in update_order:
        activation = 0
        for j in range(num_neurons):
            activation += sum_matrix[neuron_idx][j] * s[j]
        
        if activation >= 0:
            s[neuron_idx] = 1
        else:
            s[neuron_idx] = -1
            
    if s == s_old:
        break

print(f"\nPatrón reconocido final:\n{s}")