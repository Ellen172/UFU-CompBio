isTypeSetosa <- function (valor){
  if(valor == "Iris-setosa"){
    return (TRUE)
  }
  return (FALSE)
}
isTypeVersicolor <- function (valor){
  if(valor == "Iris-versicolor"){
    return (TRUE)
  }
  return (FALSE)
}


# função para cria o dataframe a ser usado no treinamento da rede neural com as posições 
# escolhidas aleatoriamente da base iris
create_dataframe_iris <- function (data_iris, pos_iris){
  matrix_data <- matrix(nrow=length(pos_iris), ncol=ncol(data_iris))
  nrow = 1;
  for (pos in pos_iris) {
    ncol = 1;
    for(column in 1:ncol(data_iris)){
      if(column==5){
        if(isTypeSetosa(data_iris[pos,column])){
          matrix_data[nrow,ncol] <- 1
        } else if(isTypeVersicolor(data_iris[pos,column])){
          matrix_data[nrow,ncol] <- -1
        }
      } else {
        matrix_data[nrow,ncol] <- data_iris[pos, column]
      }
      ncol = ncol+1
    }
    nrow = nrow+1
  }
  return (data.frame(matrix_data))
}

# função para criar um dataframe de teste com os dados da base iris que não foram usado no treinamento.
create_dataframe_test <- function(data_iris, pos_iris){
  matrix_data <- matrix(nrow=nrow(data_iris)-length(pos_iris), ncol=ncol(data_iris)+1)
  rmatrix = 1
  for(rdata in 1:nrow(data_iris)){
    if(rdata %in% pos_iris == FALSE){
      cmatrix = 1
      for(cdata in 1:(ncol(data_iris)+1)){
        if(cdata == 5){
          if(isTypeSetosa(data_iris[rdata,cdata])){
            matrix_data[rmatrix,cmatrix] <- 1
          }
          else if(isTypeVersicolor(data_iris[rdata,cdata])){
            matrix_data[rmatrix,cmatrix] <- -1
          }
        }
        else if(cdata == 6){
          matrix_data[rmatrix,cmatrix] <- as.integer(rdata)
        }
        else {
          matrix_data[rmatrix,cmatrix] <- data_iris[rdata, cdata]
        }
        cmatrix = cmatrix+1
      }
      rmatrix = rmatrix+1
    }
  }
  return (data.frame(matrix_data))
}

# Função para treinar o Perceptron
train_perceptron <- function(data, learning_rate, epochs) {
  # Inicializar pesos aleatoriamente
  weights <- runif(n = 5, min = -1, max = 1)

  for (epoch in 1:epochs) {
    for (i in 1:nrow(data)) {
      # Adicionar o viés (bias) aos dados
      x <- c(1, data[i, 1], data[i, 2], data[i, 3], data[i, 4])
      
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
      error <- data[i, 5] - predicted_class
      weights <- weights + learning_rate * error * x
    }
  }
  
  return(weights)
}

# Função para fazer previsões com o Perceptron treinado
predict_perceptron <- function(weights, x1, x2, x3, x4) {
  x <- c(1, x1, x2, x3, x4)
  z <- sum(weights * x)
  if (z > 0) {
    return(1)
  } else {
    return(-1)
  }
}

# definindo variaveis

excluir_classe = TRUE
#   excluir_classe -- se TRUE, exclui uma das classes do data.iris e executa com 2 classes 
percent_data = 0.2
#   percent_data -- define o percentual utilizado do banco data.iris para o treinamento da rede 
learning_rate = 0.01
#   learning_rate -- taxa de aprendizagem 
epochs = 200
#   epochs -- total de interações 

# execução do perceptron
  
# ler arquivo que contem os dados da base original
library(readr)
iris <- read_csv("/home/ellen/Documentos/UFU/UFU-CompBio/trabalho2/iris/iris.data", 
                 col_names = FALSE, show_col_types = FALSE)

# cria um dataframe com os dados da base iris
if(excluir_classe == TRUE){
  data_iris <- subset(data.frame(iris), iris['X5'] != "Iris-virginica")
} else {
  data_iris <- data.frame(iris)
}

# cria um vetor com as posições selecionadas para treinamento
pos_iris <- sample(1:nrow(data_iris), nrow(data_iris)*percent_data)

# cria o dataframe a ser usado no treinamento da rede neural com as posições 
# escolhidas aleatoriamente da base iris
data <- create_dataframe_iris(data_iris, pos_iris)
print(data)

# Treinar o Perceptron
trained_weights <- train_perceptron(data, learning_rate, epochs)

# criar um dataframe de teste com os dados da base iris que não foram usado no treinamento.
data_test <- create_dataframe_test(data_iris, pos_iris)
print(data_test)

# Testar o Perceptron com dados data_test
for(row_data in 1:nrow(data_test)){
  prediction <- predict_perceptron(trained_weights, data_test[row_data, 1], data_test[row_data, 2], 
                                   data_test[row_data, 3], data_test[row_data, 4])
  # Imprimir a previsão
  if (prediction == 1){
    prediction_class = "Iris-setosa"
  } else {
    prediction_class = "Iris-versicolor"
  } 
  cat(data_test[row_data, 6], prediction_class,"\n")
}
  
