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
train <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/train_u6lujuX_CVtuZ9i.csv", stringsAsFactors = T)
#Looking at the structure of caret package.
str(train)








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
#Create Data partition
#https://www.rdocumentation.org/packages/caret/versions/6.0-73/topics/createDataPartition
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
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome")







#--------------------------------------------------------------------------------------------------------
##Training models using Caret
#--------------------------------------------------------------------------------------------------------
names(getModelInfo())   # to retrive all th emodels in CARET
#We can simply apply a large number of algorithms with similar syntax
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm')

require(Matrix)
model_xgbTree<-train(x=data.matrix(trainSet[,predictors]),y=data.matrix(trainSet[,outcomeName])
                     ,method='xgbTree')








#--------------------------------------------------------------------------------------------------------
##Parameter tuning using Caret
#--------------------------------------------------------------------------------------------------------
#The resampling technique used for evaluating the performance of the model using a set of parameters in
#Caret by default is bootstrap,but it provides alternatives for using k-fold, repeated k-fold as well 
#as Leave-one-out cross validation (LOOCV) which can be specified using trainControl().  
fitControl <- trainControl(
  method = "repeatedcv", # Repeated Cross validation   
  number = 5,            # Five fold cross validation
  repeats = 5)           # Repeat five times

#To find the parameters of a model that can be tuned, you can use
modelLookup(model='gbm')
#model         parameter                   label forReg forClass probModel
#1   gbm           n.trees   # Boosting Iterations   TRUE     TRUE      TRUE
#2   gbm interaction.depth          Max Tree Depth   TRUE     TRUE      TRUE
#3   gbm         shrinkage               Shrinkage   TRUE     TRUE      TRUE
#4   gbm    n.minobsinnode Min. Terminal Node Size   TRUE     TRUE      TRUE

#Creating grid
grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),
                    n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))

# training the model. This will take some time to run
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',
                 trControl=fitControl,tuneGrid=grid)            

# summarizing the model
print(model_gbm)
max(model_gbm$results$Accuracy)
plot(model_gbm)     #with the two dimwntional plots we can realize the parameters desired


#--------------------------------------------------------------------------------------------------------
#6.2. Using tuneLength
#--------------------------------------------------------------------------------------------------------
#using tune length
  model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,
                   tuneLength=10)
print(model_gbm)
#------------------------------------------------------Output Starts-------------------------------------
#Stochastic Gradient Boosting 

#461 samples
#5 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 368, 369, 369, 368, 370, 368, ... 
#Resampling results across tuning parameters:
  
#  interaction.depth  n.trees  Accuracy   Kappa    
#1                  50      0.8017577  0.4594217
#1                 100      0.7961148  0.4528510
#1                 150      0.7943612  0.4505052
#1                 200      0.7892047  0.4390471
#1                 250      0.7857405  0.4335479
#1                 300      0.7813640  0.4266465
#1                 350      0.7835288  0.4333079
#1                 400      0.7826683  0.4316168
#1                 450      0.7813594  0.4298134
#1                 500      0.7800690  0.4256230
#2                  50      0.7900132  0.4412522
#2                 100      0.7852634  0.4342783
#2                 150      0.7839590  0.4382156
#2                 200      0.7787741  0.4264692
#2                 250      0.7757305  0.4216496
#2                 300      0.7735516  0.4191668
#2                 350      0.7714297  0.4148595
#2                 400      0.7670956  0.4078317
#2                 450      0.7688157  0.4146627
#2                 500      0.7648697  0.4071275
#3                  50      0.7935295  0.4493223
#3                 100      0.7835336  0.4366480
#3                 150      0.7748136  0.4264744
#3                 200      0.7744023  0.4252575
#3                 250      0.7726632  0.4228503
#3                 300      0.7644443  0.4089943
#3                 350      0.7627144  0.4073282
#3                 400      0.7644206  0.4142252
#3                 450      0.7592169  0.4038400
#3                 500      0.7614054  0.4086696
#4                  50      0.7949001  0.4574659
#4                 100      0.7779189  0.4317316
#4                 150      0.7722660  0.4245811
#4                 200      0.7661788  0.4155180
#4                 250      0.7653421  0.4136697
#4                 300      0.7662120  0.4193574
#4                 350      0.7614239  0.4105382
#4                 400      0.7588434  0.4056645
#4                 450      0.7597227  0.4120296
#4                 500      0.7596424  0.4106157
#5                  50      0.7852824  0.4382352
#5                 100      0.7770066  0.4327435
#5                 150      0.7726680  0.4306747
#5                 200      0.7679039  0.4230456
#5                 250      0.7626956  0.4105638
#5                 300      0.7623127  0.4151571
#5                 350      0.7614335  0.4121356
#5                 400      0.7583520  0.4071473
#5                 450      0.7592216  0.4101203
#5                 500      0.7561779  0.4055739
#6                  50      0.7878440  0.4504569
#6                 100      0.7800835  0.4424835
#6                 150      0.7700496  0.4252255
#6                 200      0.7657013  0.4156949
#6                 250      0.7635559  0.4162351
#6                 300      0.7600681  0.4086812
#6                 350      0.7639483  0.4177132
#6                 400      0.7635229  0.4198087
#6                 450      0.7652764  0.4253630
#6                 500      0.7627194  0.4197172
#7                  50      0.7843890  0.4405546
#7                 100      0.7744026  0.4330554
#7                 150      0.7644723  0.4145880
#7                 200      0.7627098  0.4146597
#7                 250      0.7596849  0.4109318
#7                 300      0.7592077  0.4132588
#7                 350      0.7579266  0.4115004
#7                 400      0.7600960  0.4176142
#7                 450      0.7622700  0.4209727
#7                 500      0.7610082  0.4186264
#8                  50      0.7805039  0.4352825
#8                 100      0.7748371  0.4340683
#8                 150      0.7683954  0.4260969
#8                 200      0.7583473  0.4025539
#8                 250      0.7561642  0.4010266
#8                 300      0.7605356  0.4125078
#8                 350      0.7548881  0.4032597
#8                 400      0.7535786  0.4017240
#8                 450      0.7579172  0.4111580
#8                 500      0.7574777  0.4083286
#9                  50      0.7809483  0.4381317
#9                 100      0.7740101  0.4310472
#9                 150      0.7718033  0.4322294
#9                 200      0.7665853  0.4234219
#9                 250      0.7648980  0.4242828
#9                 300      0.7618164  0.4194264
#9                 350      0.7618589  0.4165024
#9                 400      0.7579314  0.4076622
#9                 450      0.7561874  0.4078378
#9                 500      0.7562158  0.4088312
#10                  50      0.7774036  0.4328266
#10                 100      0.7704848  0.4268413
#10                 150      0.7700640  0.4311069
#10                 200      0.7670011  0.4270446
#10                 250      0.7578890  0.4074298
#10                 300      0.7626437  0.4205383
#10                 350      0.7573931  0.4086315
#10                 400      0.7612874  0.4188479
#10                 450      0.7587304  0.4137263
#10                 500      0.7574447  0.4095822

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
varImp(object=model_rf)
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
