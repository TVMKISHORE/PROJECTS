#Load Train Data
#To replace all the spaces and nulls as NA while loading
Train_loan <- read.csv("C:/Users/Admin/Downloads/Loan_prediction/train_u6lujuX_CVtuZ9i.csv", na.strings=c("","NA"))
Train_loan=Train_loan[complete.cases(Train_loan),]

#Load test data
#To replace all the spaces and nulls as NA while loading
Train_loan <- read.csv("C:/Users/Admin/Downloads/Loan_prediction/test_Y3wMUE5_7gLdaTN.csv", na.strings=c("","NA"))

install.packages("missForest")
library(missForest)

#install package and load library
install.packages("Hmisc")
library(Hmisc)

#adding Dummy variables 
exclude_ftrs=c("Loan_ID","Gender","Education" )


impute_arg <- aregImpute(~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width +
                           Species, data = iris.mis, n.impute = 5)