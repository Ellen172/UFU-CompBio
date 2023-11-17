from sklearn.model_selection import train_test_split # Usada para separar dados de treino e teste
import matplotlib.pyplot as plt # Usada para plotar os graficos
import numpy as np
import pandas as pd
import random 

# PASSO 1 >> Selecionar um conjunto de dados para treinamento

# Carregar os dados Iris
iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

# Pegando duas classes para classificação (Iris-setosa e Iris-versicolor)
# Removendo Iris-virginica do banco iris
new_iris_data = iris_data.iloc[0:100, 4].values
new_iris_data = np.where(new_iris_data == 'Iris-setosa', -1, 1)

# Cria array apenas com os atributos: tamanho do pétala e tamanho da sépala
final_iris_data = np.array(iris_data.iloc[0:100, [0,2]].values)
print("\nfinal_iris_data: \n",final_iris_data)

# Separando de forma aleatoria os dados de treinamento e teste
# Para alterar o tamanho desejado dos dados de treinamento basta alterar o argumento 'train_size'
train_x, test_x, train_y, test_y = train_test_split(final_iris_data, new_iris_data, train_size=0.5)

# Função para treinar o perceptron
def train_perceptron(x, y, learning_rate=0.01, epochs=10, ng=20):

    weights = [] # Armazenar os pesos
    fitness = [] # Armazenar as aptidões dos pesos
    errors_ = [] # Armazenar as taxas de erro

    # PASSO 2 >> Gerar população de pesos (matriz) com AG
    
    # Iniciando os pesos randomicamente
    print("\nPesos Iniciais: ")
    for i in x:
        weight = np.random.rand(1 + x.shape[1]) - 0.5 
        weights.append(weight)
        print(weight)
    
    # Adicionando o valor 1 em todos os dados de x para nao alterar o bias dos pesos
    x_with_bias = np.hstack((np.ones((x.shape[0], 1)), x)) 

    # PASSO 3 >> Aplicar Perceptron para classificar

    for w in weights:
        for epoch in range(epochs):
            errors = 0
            for i in range(len(y)):
                xi = x_with_bias[i]
                yi = y[i]

                # Calculando o produto escalar
                z = np.dot(w, xi)
                predicted_class = 1 if z > 0 else -1

                # Funcao de erro para atualizacao dos pesos
                error = (yi - predicted_class)
                errors += int(error != 0.0)

                # Atualizando os pesos
                w =[w[j] + learning_rate * error * xi[j] for j in range(3)]
                
            # Salvando os erros da epoca (utilizado apenas para analise grafica)
            errors_.append(errors)

    print("\nPesos pós perceptron: ")
    for w in weights:
        print(w)

    # PASSO 4 >> Evoluir pesos com AG até um criterio de parada

    # Avalia cada individuo, nesse caso, a aptidão será definida 
    # como a soma dos pesos
    print("\nFitness inicial: ")
    for w in weights:
        fit = sum(w)
        print(fit)
        fitness.append(fit)
    
    # Processo evolutivo 
    for i in range(ng):
        
        # Seleciona aleatoriamente dois individuos da população (x) para reprodução
        parents = random.sample(weights, 2)

        # Seleciona um ponto aleatorio para o crossover 
        pcross = random.randint(0,1)

        # Aplica crossover, gerando filhos
        children1 = np.concatenate((parents[0][0:pcross], parents[1][pcross:]))
        children2 = np.concatenate((parents[1][pcross:], parents[0][0:pcross]))

        # PASSO 5 >> Seleção, recombinação, mutação 
        # Recombinação aritmetica simples de 1 ponto
       
        alfa = 0.1
        pcomb1 = random.randint(0,2)
        children1[pcomb1] = alfa
        pcomb2 = random.randint(0,2)
        children2[pcomb2] = alfa

        # Calcula fitness dos filhos

        fitchild1 = sum(children1)
        fitchild2 = sum(children2)

        # Encontra os dois individuos de menor fitness na população 

        menor_valor1 = fitness[0]
        pos_menor_valor1 = 0
        menor_valor2 = fitness[1]
        pos_menor_valor2 = 1

        tam_fitness = len(fitness)
        for i in range(tam_fitness):
            valor = fitness[i]
            if valor < menor_valor1:
                menor_valor2 = menor_valor1
                pos_menor_valor2 = pos_menor_valor1
                menor_valor1 = valor
                pos_menor_valor1 = i
            elif valor < menor_valor2:
                menor_valor2 = valor
                pos_menor_valor2 = i

        # Se os novos filhos estão melhores que os dois piores individuos, faz a troca

        if fitchild1 > menor_valor1 : 
            fitness[pos_menor_valor1] = fitchild1
            weights[pos_menor_valor1] = children1
        if fitchild2 > menor_valor2 :
            fitness[pos_menor_valor2] = fitchild2
            weights[pos_menor_valor2] = children2

    print("\nFitness final: ")
    for fit in fitness: 
        print(fit)

    maior_fit = fitness[0]
    pos_maior_fit = 0
    tam_fitness = len(fitness)

    for i in range(tam_fitness):
        if fitness[i] > maior_fit:
            maior_fit = fitness[0]
            pos_maior_fit = i

    return weights[pos_maior_fit]

# Treinando o perceptron
train_weights = train_perceptron(train_x, train_y)
print("\nPesos apos treinamento: ", train_weights)

# PASSO 6 >> Usar melhor individuo (train_weights) para classificar demais dados

# Função para fazer previsões usando os pesos aprendidos
def predict(x, y, weights):
    predictions = []
    x_with_bias = np.hstack((np.ones((x.shape[0], 1)), x))
    for i in range(len(y)):
        xi = x_with_bias[i]
        z = np.dot(weights, xi)
        predicted_class = 1 if z > 0 else -1
        predictions.append(predicted_class)
    
    return np.array(predictions)

# Fazendo previsões utilizando o conjunto de teste
predictions = predict(test_x, test_y, train_weights)

# Calculando a precisão do modelo
accuracy = np.mean(predictions == test_y)
print("Precisão do modelo:", accuracy)