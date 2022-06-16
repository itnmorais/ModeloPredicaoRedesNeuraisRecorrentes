#Modelo de prediÃ§Ã£o de temperatura com redes neurais recorrentes

#InstalaÃ§Ã£o dos pacotes Utilizados

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

data("weatherAUS")

data <- weatherAUS[1:3040, 14:15] #Prevendo umidade 3pm com babse na umidade 9am
summary(data)

#PrÃ©-Processamento

data_cleaned <- na.omit(data) #remover NAs

data_used = data_cleaned[1:3000,] #ObservaÃ§Ãµes a serem utilizadas

x = data_used[,1] #VariÃ¡vel explicativa 
y = data_used[,2] #VariÃ¡vel target

#Normalizando variÃ¡veis para diminuiÃ§Ã£o dos ruÃ­dos

yscaled = (y - min(y)) / (max(y) - min(y))
xscaled = (x - min(x)) / (max(x) - min(x))

y <- yscaled
x <- xscaled

#Transformando x  e y em matriz

x <- as.matrix(x)
y <- as.matrix(y)

#Particionando as observaÃ§Ãµes em sÃ©ries
X=matrix(x, nrow = 30)
Y=matrix(y, nrow = 30)

#train test split

train <- 1:80
test <- 81:100

#Criando o modelo
set.seed(12)
model <- trainr(Y = Y[,train],
                X = X[,train],
                learningrate = 0.01,
                hidden_dim = 15,
                network_type = "rnn",
                numepochs = 100)


#Plotando erro em funÃ§Ã£o do nÃºmero de Ã©pocas
model$error
plot(colMeans(model$error),type='l',xlab='epoch',ylab='errors')

#Realizando a prediÃ§Ã£o
Yp <- predictr(model,X[,test])

#Retornando Y test para uma Ãºnica coluna
Ytest <- matrix(Y[,test], nrow = 1)
Ytest <- t(Ytest)

Ypredicted <- matrix(Yp,nrow = 1)
Ypredicted <- t(Ypredicted)

#Criando um dataframe com as duas colunas obtidas

result_data <- data.frame(Ytest)
result_data$Ypredicted <- Ypredicted


#plot
plot(as.vector(t(result_data$Ytest)), col = 'red', type='l',
     main = "Actual vs Predicted Humidity: testing set",
     ylab = "Y,Yp")
lines(as.vector(t(Yp)), type = 'l', col = 'black')
legend("bottomright", c("Predicted", "Actual"),
       col = c("red","black"),
       lty = c(1,1), lwd = c(1,1))

#Percentual de variaÃ§Ã£o de uma variÃ¡vel explicada por outra


rsq <- function(y_actual,y_predict)
{
  cor(y_actual,y_predict)^2
}

rsq(result_data$Ytest,result_data$Ypredicted)

mean(result_data$Ytest)
mean(result_data$Ypredicted)
