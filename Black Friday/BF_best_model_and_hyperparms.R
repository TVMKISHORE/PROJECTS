#--------------------- Complete usage of CART package------------------------------------
##Pre-processing using Caret
##Splitting the data using Caret
##Feature selection using Caret
##Training models using Caret
##Parameter tuning using Caret
##Variable importance estimation using Caret
##Making predictions using Caret

#----------------------------------------------------------------------------------------
#useful resources 
#https://www.analyticsvidhya.com/blog/2016/12/practical-guide-to-implement-machine-learning-with-caret-package-in-r-with-practice-problem/
#http://topepo.github.io/caret/using-your-own-model-in-train.html#introduction-1
#----------------------------------------------------------------------------------------
install.packages("ggplot2")       #Dependents for CARET
install.packages("scales")        #Dependents for CARET
install.packages("ModelMetrics")  #Dependents for CARET
library("ModelMetrics")           #Dependents for CARET
library("ggplot2")                #Dependents for CARET 
install.packages("caret", dependencies = c("Depends", "Suggests"))


#Loading caret package
library("caret")

#Loading training data
train <- read.csv("C:/Users/Admin/Downloads/Black Friday/train_oSwQCTC/train.csv",stringsAsFactors = T)
test <- read.csv("C:/Users/Admin/Downloads/Black Friday/test_HujdGe7/test.csv",stringsAsFactors = T)
#train['Product_ID_num']=substr(train$Product_ID,start=2,stop=9)
trainSet$Product_ID_num <- NULL
#train$Product_ID_num=as.numeric(train$Product_ID_num)
#library(dplyr)
#train<-train%>% arrange(Product_ID_num) 

#Just conside imp features from Boruta package
#[1] "Product_ID"         "Gender"             "Age"                "Occupation"        
#[5] "City_Category"      "Product_Category_1" "Product_Category_2" "Product_Category_3"

#train['Product_ID_num']=substr(train$Product_ID,start=4,stop=7)
train$Stay_In_Current_City_Years <- NULL
train$Marital_Status <-NULL
train$User_ID <- NULL
train$Product_ID <- NULL



#----------------------------------------
# Random sample above data set
#----------------------------------------
library(caret)
index <- createDataPartition(X_train$Purchase, p=1, list=FALSE)
train_random_samp <- X_train[ index,]
#train_random_samp <- train[1:1152,]
#train_random_samp <- train

y <- train_random_samp$Purchase
X <- subset(train_random_samp, select=-c(Purchase,Product_ID,flag_high))

#----------------------------------------
#Impute missing data and transforming
#----------------------------------------
train_random_samp$Product_Category_2[is.na(train_random_samp$Product_Category_2)] <- 99
train_random_samp$Product_Category_3[is.na(train_random_samp$Product_Category_3)] <- 99

train_random_samp$Occupation=as.factor(train_random_samp$Occupation)
train_random_samp$Product_Category_1=as.factor(train_random_samp$Product_Category_1)
train_random_samp$Product_Category_2=as.factor(train_random_samp$Product_Category_2)
train_random_samp$Product_Category_3=as.factor(train_random_samp$Product_Category_3)
train_random_samp$Product_ID_num=as.numeric(train_random_samp$Product_ID_num)

trainSet=train_random_samp

#--------------------------------------------------------------------------------------------------------
##Seperate IV and DV
#--------------------------------------------------------------------------------------------------------
#seperate target/outcome variable field and Convert into numeric 
#Converting every categorical variable to numerical using dummy variables
dmy <- dummyVars(" ~ .", data = trainSet,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = trainSet))
trainSet <- train_transformed





#--------------------------------------------------------------------------------------------------------
##Feature selection using Caret
#--------------------------------------------------------------------------------------------------------
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Purchase'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                         rfeControl = control)
#The top 5 variables (out of 80):
#Product_Category_1.5, 
#Product_Category_1.8, 
#Product_Category_1.11, 
#Product_Category_1.4, 
#Product_ID_num

#-----------------------------------------------------------------------------------------------
# New predictors considered
#-----------------------------------------------------------------------------------------------
predictors <- 
  c("Product_Category_1.5", 
    "Product_Category_1.8", 
    "Product_Category_1.11", 
    "Product_Category_1.4", 
    "Product_ID_num")


#--------------------------------------------------------------------------------------------------------
##Training models using Caret
#--------------------------------------------------------------------------------------------------------
names(getModelInfo())   # to retrive all th emodels in CARET
#We can simply apply a large number of algorithms with similar syntax
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')

#shrinkage interaction.depth n.minobsinnode n.trees     RMSE  Rsquared   RMSESD RsquaredSD
#1       0.1                 1             10      50 4289.065 0.3391840 256.4979 0.04985724
#4       0.1                 2             10      50 4096.081 0.3793997 279.2178 0.05152050
#7       0.1                 3             10      50 4010.333 0.3989443 281.6961 0.04797185
#2       0.1                 1             10     100 4111.902 0.3725835 288.3649 0.05283989
#5       0.1                 2             10     100 4024.221 0.3913690 292.6460 0.05023104
#8       0.1                 3             10     100 3997.082 0.4003332 301.2130 0.04939442
#3       0.1                 1             10     150 4051.713 0.3839760 289.9980 0.05014477
#6       0.1                 2             10     150 4005.270 0.3972300 296.0111 0.04822953
#9       0.1                 3             10     150 4014.232 0.3976933 298.0284 0.04843895

#model_gbm$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
#8     100                 3       0.1             10

model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
# Too high RMSE
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
# Too high RMSE
model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm')
#parameter     RMSE  Rsquared   RMSESD RsquaredSD
#1      none 3390.649 0.5838161 169.6665 0.03315018

require(Matrix)

BMS_xgbTree<-train(trainSet[,predictors],trainSet[,outcomeName],
                   method='xgbTree',)

BMS_xgbTree<-train(X,y,method='xgbTree')

#    eta max_depth gamma colsample_bytree min_child_weight    subsample    nrounds     RMSE      Rsquared
#72  0.4         1     0              0.8                1      1.00        150   3297.975   0.5885457
#BMS_xgbTree$bestTune
#72     150         1 0.4     0              0.8                1         1




#--------------------------------------------------------------------------------------------------------
##Parameter tuning using Caret
#--------------------------------------------------------------------------------------------------------
#The resampling technique used for evaluating the performance of the model using a set of parameters in
#Caret by default is bootstrap,but it provides alternatives for using k-fold, repeated k-fold as well 
#as Leave-one-out cross validation (LOOCV) which can be specified using trainControl().  
fitControl <- trainControl(
  method = "repeatedcv", # Repeated Cross validation   
  number = 5,            # Five fold cross validation
  repeats = 1)           # Repeat five times

#To find the parameters of a model that can be tuned, you can use
modelLookup(model='gbm')
#model         parameter                   label forReg forClass probModel
#1   gbm           n.trees   # Boosting Iterations   TRUE     TRUE      TRUE
#2   gbm interaction.depth          Max Tree Depth   TRUE     TRUE      TRUE
#3   gbm         shrinkage               Shrinkage   TRUE     TRUE      TRUE
#4   gbm    n.minobsinnode Min. Terminal Node Size   TRUE     TRUE      TRUE

#Creating grid

#    eta max_depth gamma colsample_bytree min_child_weight    subsample    nrounds     RMSE      Rsquared
#72  0.4         1     0              0.8                1      1.00        150   3297.975   0.5885457
#--------------------------First tune
grid <- expand.grid(eta=c(.4,.2,.1),
                    max_depth=c(6,4,1),
                    colsample_bytree = c(.3,.6,.8),
                    min_child_weight=c(1,2,3),
                    subsample=1.00,
                    gamma=0,
                    nrounds=c(150))
#BMS_xgbTree$results
#     eta    max_depth gamma    colsample_bytree   min_child_weight subsample nrounds   RMSE      Rsquared
#12   0.1         4     0              0.3                3         1         150       2379.295  0.5179405
#BMS_xgbTree$bestTune
#    nrounds   max_depth eta gamma    colsample_bytree   min_child_weight subsample
#12     150         4    0.1     0              0.3                3         1

#--------------------------Second tune
grid <- expand.grid(eta=c(.1,.08,.06,.03),
                    max_depth=4,
                    colsample_bytree =.3,
                    min_child_weight=c(3,4,5),
                    subsample=1.00,
                    gamma=0,
                    nrounds=c(150,300,450))

#BMS_xgbTree$results
#   eta     max_depth gamma colsample_bytree min_child_weight subsample  nrounds      RMSE     Rsquared
#5  0.03         4     0            0.3               4         1        300        2363.910   0.5229528
#BMS_xgbTree$bestTune
#    nrounds     max_depth  eta   gamma   colsample_bytree   min_child_weight subsample
#5     300         4        0.03     0              0.3                4         1

#--------------------------third tune
grid <- expand.grid(eta=c(.03,.01,.008,.006,.004,.002),
                    max_depth=4,
                    colsample_bytree =.3,
                    min_child_weight=4,
                    subsample=1.00,
                    gamma=0,
                    nrounds=c(300))

#BMS_xgbTree$results
#    eta      max_depth gamma colsample_bytree min_child_weight subsample  nrounds      RMSE     Rsquared
#16 0.030         4     0            0.3                4         1         325         2381.592
#BMS_xgbTree$bestTune
#     nrounds      max_depth  eta   gamma   colsample_bytree  min_child_weight subsample
#16     325         4         0.03     0              0.3                4         1
#--------------------------Fourth tune with 5 folds
grid <- expand.grid(eta=.03,
                    max_depth=4,
                    colsample_bytree =.3,
                    min_child_weight=4,
                    subsample=1.00,
                    gamma=0,
                    nrounds=c(350))    # increases with observations


#BMS_xgbTree$bestTune
#eta max_depth colsample_bytree min_child_weight subsample gamma nrounds    RMSE  Rsquared
#1 0.03         4              0.3                4         1     0     300 2371.32 0.5205529

#RMSESD RsquaredSD
#1 99.05739 0.03736402
#nrounds max_depth  eta gamma colsample_bytree min_child_weight subsample
#1     300         4 0.03     0              0.3                4         1

# training the model. This will take some time to run
BMS_xgbTree<-train(trainSet[,predictors],trainSet[,outcomeName],method='xgbTree',trControl=fitControl,
                 tuneGrid=grid,verbose=TRUE)
BMS_xgbTree<-train(X,y,method='xgbTree')

library(caret)
BMS_xgbTree<-train(X_train,y,method='xgbTree')

save(BMS_xgbTree,file="BMS_xgbTree.Rdata")
# summarizing the model
print(model_gbm)
max(model_gbm$results$Accuracy)
plot(model_gbm)     #with the two dimwntional plots we can realize the parameters desired


#--------------------------------------------------------------------------------------------------------
#6.2. Using tuneLength
#--------------------------------------------------------------------------------------------------------
grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),
                    n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))
#using tune length
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,
                 tuneLength=10)
print(model_gbm)
#------------------------------------------------------Output Starts-------------------------------------

#Tuning parameter 'shrinkage' was held constant at a value of 0.1
#Tuning parameter 'n.minobsinnode' was
#held constant at a value of 10
#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were n.trees = 50, interaction.depth = 1, shrinkage = 0.1
#and n.minobsinnode = 10. 
#---------------------------------------------------------Output Ends-------------------------------
plot(model_gbm)   #to plot a graph and show the hyperameter trend




















#--------------------------------------------------------------------------------------------------------
#Variable importance estimation using caret
#--------------------------------------------------------------------------------------------------------
#Checking variable importance for GBM

#Variable Importance
varImp(object=model_gbm)
#gbm variable importance
#Overall
#Credit_History    100.000
#LoanAmount         16.633
#ApplicantIncome     7.104
#CoapplicantIncome   6.773
#Loan_Amount_Term    0.000
#Plotting Varianle importance for GBM
plot(varImp(object=model_gbm),main="GBM - Variable Importance")



#Checking variable importance for RF
.
#rf variable importance
#Overall
#Credit_History     100.00
#ApplicantIncome     73.46
#LoanAmount          60.59
#CoapplicantIncome   40.43
#Loan_Amount_Term     0.00
#Plotting Varianle importance for Random Forest
plot(varImp(object=model_rf),main="RF - Variable Importance")


#Checking variable importance for NNET
varImp(object=model_nnet)
#nnet variable importance
#Overall
#ApplicantIncome    100.00
#LoanAmount          82.87
#CoapplicantIncome   56.92
#Credit_History      41.11
#Loan_Amount_Term     0.00
#Plotting Variable importance for Neural Network
plot(varImp(object=model_nnet),main="NNET - Variable Importance")



#Checking variable importance for GLM
varImp(object=model_glm)
#glm variable importance
#Overall
#Credit_History    100.000
#CoapplicantIncome  17.218
#Loan_Amount_Term   12.988
#LoanAmount          5.632
#ApplicantIncome     0.000
#Plotting Variable importance for GLM
plot(varImp(object=model_glm),main="GLM - Variable Importance")



#Checking variable importance for GLM
varImp(object=model_xgbTree)
#xgbTree variable importance
#Overall
#Credit_History     100.00
#LoanAmount          27.85
#ApplicantIncome     10.66
#CoapplicantIncome    9.54
#Loan_Amount_Term     0.00
plot(varImp(object=model_xgbTree),main="XGB - Variable Importance")

#Predictors that are important for the majority of models represents genuinely important predictors.
#we should use predictions from models that have significantly different variable importance as their
#predictions are also expected to be different.










#--------------------------------------------------------------------------------------------------------
##Making predictions & Validation using Caret
#--------------------------------------------------------------------------------------------------------
#Predictions
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
table(predictions)
#predictions
#0   1
#28 125
confusionMatrix(predictions,testSet[,outcomeName])
