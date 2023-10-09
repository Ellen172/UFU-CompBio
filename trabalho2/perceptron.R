# Gera dados de treinamento
set.seed(123) # Para replicação dos experimetos; remover para testar parâmetros
x1 <- runif(n = 100, min = 0, max = 10)
x2 <- runif(n = 100, min = 0, max = 10)
class <- ifelse(x1 + x2 > 10, 1, -1)

# Cria um dataframe com os dados de treinamento
data <- data.frame(x1, x2, class)

# Função para treinar o Perceptron
train_perceptron <- function(data, learning_rate = 0.01, epochs = 100) {
  # Inicializar pesos aleatoriamente
  weights <- runif(n = 3, min = -1, max = 1)
  #weights <- c(-3,0.3,0.3)
  
  for (epoch in 1:epochs) {
    for (i in 1:nrow(data)) {
      # Adicionar o viés (bias) aos dados
      x <- c(1, data[i, "x1"], data[i, "x2"])
      
      # Calcular o produto escalar entre os pesos e os dados de entrada
      z <- sum(weights * x)
      
      # Aplicar a função de ativação (função degrau)
      # sigma
      if (z > 0) {
        predicted_class <- 1
      } else {
        predicted_class <- -1
      }
      
      # Atualizar os pesos com base no erro
      error <- data[i, "class"] - predicted_class
      weights <- weights + learning_rate * error * x
    }
  }
  
  return(weights)
}

# Treinar o Perceptron
trained_weights <- train_perceptron(data)
print(trained_weights)

# Função para fazer previsões com o Perceptron treinado
predict_perceptron <- function(weights, x1, x2) {
  x <- c(1, x1, x2)
  z <- sum(weights * x)
  if (z > 0) {
    return(1)
  } else {
    return(-1)
  }
}


# Testar o Perceptron com novos dados
new_x1 <- 6
new_x2 <- 5
prediction <- predict_perceptron(trained_weights, new_x1, new_x2)

# Imprimir a previsão
if (prediction == 1) {
  cat("A classe prevista é 1\n")
} else {
  cat("A classe prevista é -1\n")
}
