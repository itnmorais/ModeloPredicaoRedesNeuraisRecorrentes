#MODELO PARA PREDIÇÃO DE TEMPERATURA COM REDES NEURAIS RECORRENTES (RNN)

#01 - Instalação e carregamento dos pacotes utilizados

pacotes <- c("rattle","rnn","ggplot2","dplyr")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}

library("rattle")
library("rnn")
library("ggplot2")

#02 - Carregando o dataset weatherAUS

data("weatherAUS")

#03 - Objetivo do modelo: Prever a temperatura às 3pm com base na temperatura de 9am
#Carragando os dados da cidade de Albury
data <- weatherAUS[1:3680,20:21]
summary(data)

#04 - Pré-processamento
data_cleaned <- na.omit(data) #Removendo NA's
data_used <- data_cleaned[1:3660,] #Observações a serem utilizadas

x = data_used[,1] #variável explicativa
y = data_used[,2] #variável target

# 05 - Normalizando variáveis para diminuição dos "ruídos"
xscaled <- (x - min(x))/(max(x) - min(x))
yscaled <- (y- min(y))/ (max(y) - min(y))

x <- xscaled
y <- yscaled

#06 - Transformando x e y em matriz
x <- as.matrix(x)
y <- as.matrix(y)

#07 - Particionando as observações em séries
x = matrix(x, nrow = 30)
y = matrix(y, nrow = 30)

#08 - Separando as bases de treino e teste
train <- 1:98
test <- 99:122

#09 - Criando o modelo
set.seed(12)
model <- trainr(Y = y[,train],
                X = x[,train],
                learningrate = 0.01,
                hidden_dim = 15,
                network_type = "rnn",
                numepochs = 100)

#10 - Plotando erro em função do número de épocas
model$error
plot(colMeans(model$error),type='l', xlab = 'epoch', ylab = 'errors')

#11 - Realizando a predição
yp <- predictr(model,x[,test])

#12 - Retornando y test para uma única coluna
ytest <- matrix(y[,test], nrow = 1)
ytest <- t(ytest)

ypredicted <- matrix(yp,nrow = 1)
ypredicted <- t(ypredicted)

#13 - Criando um dateframe com as duas colunas obtidas  na base de teste
result_data <- data.frame(ytest)
result_data$ypredicted <- ypredicted

result_data <- rename(result_data,
                      Temp_real = 1,
                      Temp_predita = 2)
