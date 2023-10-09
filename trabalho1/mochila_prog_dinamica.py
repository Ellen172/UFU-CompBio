import numpy as np
import re
import time

# função para 
# parametros: 
# size -> numero de itens
# value -> array com o valor de cada item
# weight -> array com o peso de cada item
# capacity -> capacidade da mochila 
# dp -> matriz 'n' linhas e 'capacity+1' colunas
def knapsack(size, value, weight, capacity, dp):
    if size == 0 or capacity == 0:
        return 0
    if dp[size - 1][capacity] != -1:
        return dp[size - 1][capacity]
    if weight[size - 1] > capacity:
        dp[size - 1][capacity] = knapsack(size - 1, value, weight, capacity, dp)
        return dp[size - 1][capacity]
    a = value[size - 1] + knapsack(size - 1, value, weight, capacity - weight[size - 1], dp)
    b = knapsack(size - 1, value, weight, capacity, dp)
    dp[size - 1][capacity] = max(a, b)
    return dp[size - 1][capacity]

# os arquivos de entrada contem: 
#   n -> numero de entradas
#   capacity -> capacidade da mochila 
# para cada entrada: 
#   id -> utilizado para diferenciar o item dos demais
#   value -> valor do item
#   weight -> peso do item 
def solve_knapsack_problem(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    n = int(lines[0].strip())
    capacity = int(lines[-1].strip())
    
    id, value, weight = [], [], []
    for line in lines[1:-1]:
        numbers = re.findall(r"[0-9]+", line)
        id.append(int(numbers[0]) - 1)
        value.append(int(numbers[1]))
        weight.append(int(numbers[2]))
    
    # cria uma matriz com 'n' linhas e 'capacity+1' colunas
    # preenche com o valor -1
    dp = np.full((n, capacity + 1), -1, dtype=int)

    print("n = ", n, "\t capacity = ", capacity, "\n")
    print("value = ", value, "\n")
    print("weight = ", weight, "\n")
    print("dp:")
    print(dp)
    max_value = knapsack(n, value, weight, capacity, dp)
    return max_value

# função principal
def main():
    output_max_values = []

    # chama a função solve_knapsack_problem para as entradas definidas nos arquivos da pasta input
    for iterator in range(1, 3):
        print("\n\nInstancia ", iterator, "\n")
        input_file_path = f"/home/ellen/Documentos/UFU/UFU-CompBio/trabalho1/input/input{iterator}.in"
        max_value = solve_knapsack_problem(input_file_path)
        output_max_values.append(max_value)
        output_line = f"Instancia {iterator} : {max_value}\n"
        
        with open("/home/ellen/Documentos/UFU/UFU-CompBio/trabalho1/output/dynamic.out", "a+") as output_file:
            output_file.write(output_line)

if __name__ == "__main__":
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time} seconds")
