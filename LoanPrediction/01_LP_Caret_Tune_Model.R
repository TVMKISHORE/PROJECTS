#--------------------------------------------------------------------------------------------------------
#http://topepo.github.io/caret/using-your-own-model-in-train.html#introduction-1
#--------------------------------------------------------------------------------------------------------
install.packages("ggplot2")       #Dependents for CARET
install.packages("scales")        #Dependents for CARET
install.packages("ModelMetrics")  #Dependents for CARET
install.packages("caret", dependencies = c("Depends", "Suggests"))
library("ModelMetrics")           #Dependents for CARET
library("ggplot2")                #Dependents for CARET 
library("caret")

#--------------------------------------------------------------------------------------------------------
#preprocessing,Varible selection,tuning,training,Validation using CARET
#--------------------------------------------------------------------------------------------------------

train <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/train_u6lujuX_CVtuZ9i.csv", stringsAsFactors = T)

#--------------------------------------------------------------------------------------------------------
##Pre-processing using Caret
#--------------------------------------------------------------------------------------------------------
sum(is.na(train))   # to know the total NA observations 
#use Caret to impute these missing values using KNN algorithm
preProcValues <- preProcess(train, method = c("knnImpute","center","scale"))
library('RANN')
train_processed <- predict(preProcValues, train)
sum(is.na(train_processed))

# seperate target/outcome variable field and Convert into numeric 
train_processed$Loan_Status<-ifelse(train_processed$Loan_Status=='N',0,1)
id<-train_processed$Loan_ID
train_processed$Loan_ID<-NULL

#Converting every categorical variable to numerical using dummy variables
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))

#Converting the dependent variable back to categorical
train_transformed$Loan_Status<-as.factor(train_transformed$Loan_Status)
#--------------------------------------------------------------------------------------------------------
##Splitting the data using Caret
#--------------------------------------------------------------------------------------------------------
#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(train_transformed$Loan_Status, p=0.75, list=FALSE)
trainSet <- train_transformed[ index,]
testSet <- train_transformed[-index,]
#--------------------------------------------------------------------------------------------------------
##Feature selection using Caret
#--------------------------------------------------------------------------------------------------------
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Loan_Status'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                         rfeControl = control)
Loan_Pred_Profile
#Recursive feature selection
#Outer resampling method: Cross-Validated (10 fold, repeated 3 times)
#Resampling performance over subset size:
#  Variables Accuracy  Kappa AccuracySD KappaSD Selected
#4   0.7737 0.4127    0.03707 0.09962        
#8   0.7874 0.4317    0.03833 0.11168        
#16   0.7903 0.4527    0.04159 0.11526        
#18   0.7882 0.4431    0.03615 0.10812        
#The top 5 variables (out of 16):
#  Credit_History, LoanAmount, Loan_Amount_Term, ApplicantIncome, CoapplicantIncome
#Taking only the top 5 predictors
predictors<-c("Credit_History", "LoanAmount","ApplicantIncome", "CoapplicantIncome")

#By creating models below models and retriving varImp we came to know that 'Loan_Amount_Term' is not important
#model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
#model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
#model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
#model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm')










#--------------------------------------------------------------------------------------------------------
##Parameter tuning using Caret--Tuning XGB
#--------------------------------------------------------------------------------------------------------
fitControl <- trainControl(
  method = "repeatedcv", # Repeated Cross validation   
  number = 5,            # Five fold cross validation
  repeats = 3)           # Repeat five times
#--------------------------------------------------------------------------------------------------------
#6.2. Using tuneLength
#--------------------------------------------------------------------------------------------------------
#using tune length
model_xgb<-train(data.matrix(trainSet[,predictors]),data.matrix(trainSet[,outcomeName]),method='xgbTree',trControl=fitControl,
                 tuneLength=10)
print(model_xgb)
#-------------------------------------Output
#0.3   8         0.6               0.6111111  250      0.7316963  0.3445041
#0.3   8         0.6               0.6111111  300      0.7317044  0.3431910
#0.3   8         0.6               0.6111111  350      0.7295695  0.3397887
#0.3   8         0.6               0.6111111  400      0.7266553  0.3336862
#Tuning parameter 'gamma' was held constant at a value of 0
#Tuning parameter 'min_child_weight' was
#held constant at a value of 1
#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were nrounds = 50, max_depth = 1, eta = 0.3, gamma = 0,
#colsample_bytree = 0.8, min_child_weight = 1 and subsample = 0.6111111. 
#-------------------------------------Output
plot(model_xgb)
save(model_xgb,file="model_xgb.Rdata")
load("model_xgb.Rdata")
model_xgb$bestTun
#-------------------------------------------------------Output
#nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
#121      50         1 0.3     0              0.8                1 0.6111111
#-------------------------------------------------------Output
#--------------------------------------------------------------------------------------------------------
##Parameter tuning using Caret--Tuning GBM
#--------------------------------------------------------------------------------------------------------
fitControl <- trainControl(
  method = "repeatedcv", # Repeated Cross validation   
  number = 5,            # Five fold cross validation
  repeats = 5)           # Repeat five times

grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),
                    n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))

model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',
                 trControl=fitControl,tuneGrid=grid)   
plot(model_gbm)
save(model_gbm,file="model_gbm.Rdata")
load("model_gbm.Rdata")
model_gbm
#-------------------------------------------------------Output
#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were n.trees = 10, interaction.depth = 1, shrinkage = 0.05
#and n.minobsinnode = 3. 
#--------------------------------------------------------------------------------------------------------
##Parameter tuning using Caret--Tuning RF
#--------------------------------------------------------------------------------------------------------
names(getModelInfo())   # to retrive all th emodels in CARET
modelLookup(model='rf')
model_rf<-train(data.matrix(trainSet[,predictors]),data.matrix(trainSet[,outcomeName]),method='rf',trControl=fitControl,
                 tuneLength=10)
plot(model_rf)
save(model_rf,file="model_rf.Rdata")
load("model_rf.Rdata")
model_rf
#-------------------------------------------------------Output
#Random Forest 

#461 samples
#4 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 369, 369, 369, 368, 369, 368, ... 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#2     0.7630248  0.3846590
#3     0.7552029  0.3723471
#4     0.7508593  0.3622519

#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was mtry = 2. 
#-------------------------------------------------------Output
#--------------------------------------------------------------------------------------------------------
##Parameter tuning using Caret--Tuning nnet(Nural Networks)
#--------------------------------------------------------------------------------------------------------
model_nnet<-train(data.matrix(trainSet[,predictors]),data.matrix(trainSet[,outcomeName]),method='nnet',trControl=fitControl,
                tuneLength=10)
load("model_nnet.Rdata")
save(model_nnet,file="model_nnet.Rdata")
plot(model_rf)  #model_nnet was named as model_rf by mistake






#-------------------------------------------------------------------------------------------------------
#The best model is GBM so lets build a model now with the tuned perameters 
#-------------------------------------------------------------------------------------------------------
test <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/test_Y3wMUE5_7gLdaTN.csv", stringsAsFactors = T)
#--------------------------------------------------------------------------------------------------------
##Pre-processing using Caret
#--------------------------------------------------------------------------------------------------------
sum(is.na(test))   # to know the total NA observations 
#use Caret to impute these missing values using KNN algorithm
preProcValues <- preProcess(test, method = c("knnImpute","center","scale"))
library('RANN')
test_processed <- predict(preProcValues, test)
sum(is.na(test_processed))


#Converting every categorical variable to numerical using dummy variables
test_processed$Loan_ID<-NULL
dmy <- dummyVars(" ~ .", data = test_processed,fullRank = T)
test_transformed <- data.frame(predict(dmy, newdata = test_processed))
predictors<-c("Credit_History", "LoanAmount","ApplicantIncome", "CoapplicantIncome")

#The final values used for the model were n.trees = 10, interaction.depth = 1, shrinkage = 0.05
#and n.minobsinnode = 3. 

install.packages('gbm')
library('gbm')
gbm_model_final<- gbm( Loan_Status ~ Credit_History+LoanAmount+ApplicantIncome+CoapplicantIncome,
                       data=train_processed, 
                       distribution="bernoulli",   #This is what used in the above CV model (model_gbm$finalModel)
                       n.trees = 10,
                       interaction.depth = 1,
                       n.minobsinnode = 3,
                       shrinkage = 0.05
        )



save(gbm_model_final,file="gbm_model_final.Rdata")
load('gbm_model_final.Rdata')


#---------------------------------------------------------------------------------------------------
# Generate LB Score with filan model
#---------------------------------------------------------------------------------------------------
y_pred<- predict(gbm_model_final,test_transformed[,predictors],n.trees = 10,type="link")
test_result =as.data.frame(test["Loan_ID"])
y_pred<- as.data.frame(y_pred)
test_result['Loan_Status'] = as.character(y_pred$y_pred)
Loan_file=test_result
Loan_file$Loan_Status[Loan_file$Loan_Status==1.00213505847377] <- "Y"
Loan_file$Loan_Status[Loan_file$Loan_Status==0.785989213515357] <- "Y"
Loan_file$Loan_Status[Loan_file$Loan_Status==-0.224335381813572] <- "N"
write.csv(Loan_file,file='Result_GBM.csv')









#---------------------------------------------------------------------------------------------------
# Generate LB Score with CV models
#---------------------------------------------------------------------------------------------------
y_pred<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
y_pred<- predict(model_gbm,data.matrix(test_transformed[,predictors]))

test_result =as.data.frame(test["Loan_ID"])
y_pred<- as.data.frame(y_pred)
test_result['Loan_Status'] = as.character(y_pred$y_pred)
Loan_file=test_result
Loan_file$Loan_Status[Loan_file$Loan_Status==1] <- "Y"
Loan_file$Loan_Status[Loan_file$Loan_Status==0] <- "N"

write.csv(Loan_file,file='Result_GBM.csv')
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------







