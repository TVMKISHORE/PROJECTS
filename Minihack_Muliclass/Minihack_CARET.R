#----------------------------------------------------------------------------------------
#Useful resources 
#https://www.analyticsvidhya.com/blog/2016/12/practical-guide-to-implement-machine-learning-with-caret-package-in-r-with-practice-problem/
#http://topepo.github.io/caret/using-your-own-model-in-train.html#introduction-1

#XGBoost parm Tuning guide
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
#GBM Parm Tune Guide
#https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
#----------------------------------------------------------------------------------------

library("ggplot2")       #Dependents for CARET
library("scales")        #Dependents for CARET
#install.packages('ModelMetrics')
library("ModelMetrics")  #Dependents for CARET
library("ggplot2")    
#install.packages('ModelMetrics')
#install.packages("caret", dependencies = c("Depends", "Suggests"))
library("caret")
library("RANN")

test <- read.csv("test_XaoFywY.csv",stringsAsFactors=F)
train <- read.csv("train_63qYitG.csv",stringsAsFactors=F)

library(ggplot2)
sum(is.na(train))

#look for levels and replace with numbers in train set

train$Type_of_Cab[train$Type_of_Cab=="A"]<-"1"
train$Type_of_Cab[train$Type_of_Cab=="B"]<-"2"
train$Type_of_Cab[train$Type_of_Cab=="C"]<-"3"
train$Type_of_Cab[train$Type_of_Cab=="D"]<-"4"
train$Type_of_Cab[train$Type_of_Cab=="E"]<-"5"

train$Type_of_Cab=as.numeric(train$Type_of_Cab)

train$Confidence_Life_Style_Index[train$Confidence_Life_Style_Index=="A"]<-"11"
train$Confidence_Life_Style_Index[train$Confidence_Life_Style_Index=="B"]<-"22"
train$Confidence_Life_Style_Index[train$Confidence_Life_Style_Index=="C"]<-"33"

train$Confidence_Life_Style_Index=as.numeric(train$Confidence_Life_Style_Index)

train$Destination_Type[train$Destination_Type=="A"]<-"5"
train$Destination_Type[train$Destination_Type=="B"]<-"10"
train$Destination_Type[train$Destination_Type=="C"]<-"15"
train$Destination_Type[train$Destination_Type=="D"]<-"20"
train$Destination_Type[train$Destination_Type=="E"]<-"25"
train$Destination_Type[train$Destination_Type=="F"]<-"30"
train$Destination_Type[train$Destination_Type=="G"]<-"35"
train$Destination_Type[train$Destination_Type=="H"]<-"40"
train$Destination_Type[train$Destination_Type=="I"]<-"45"
train$Destination_Type[train$Destination_Type=="J"]<-"50"
train$Destination_Type[train$Destination_Type=="K"]<-"55"
train$Destination_Type[train$Destination_Type=="L"]<-"60"
train$Destination_Type[train$Destination_Type=="M"]<-"65"
train$Destination_Type[train$Destination_Type=="N"]<-"70"
  
train$Destination_Type=as.numeric(train$Destination_Type)

train$Gender<-ifelse(train$Gender=="Male",1,0)


train$Type_of_Cab[is.na(train$Type_of_Cab)]<-"6"



library(caTools)

train1=subset(train,train$Surge_Pricing_Type==1)
train2=subset(train,train$Surge_Pricing_Type==2)
train3=subset(train,train$Surge_Pricing_Type==3)



#s a m p l i n g


#-----------------------------------------------------------------------------------------------
#Splitting based on ration
#-----------------------------------------------------------------------------------------------
set.seed(213)
split = sample.split(train1$Surge_Pricing_Type, SplitRatio = 0.0037)
train_sample1 = subset(train1, split == TRUE)
split = sample.split(train2$Surge_Pricing_Type, SplitRatio = 0.0018)
train_sample2 = subset(train2, split == TRUE)
split = sample.split(train3$Surge_Pricing_Type, SplitRatio = 0.0021)
train_sample3 = subset(train3, split == TRUE)

#-----------------------------------------------------------------------------------------------
#Splitting based on ratio CARET
#-----------------------------------------------------------------------------------------------
#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(train$Surge_Pricing_Type, p=0.75, list=FALSE)
trainSet <- train[ index,]
testSet <- train[-index,]
#-----------------------------------------------------------------------------------------------








#-----------------------------------------------------------------------------------------------
# I M P U T A T I O N 
#-----------------------------------------------------------------------------------------------
train_processed=rbind(train_sample1,train_sample2,train_sample3)
train_processed$Surge_Pricing_Type=as.factor(train_processed$Surge_Pricing_Type)
train_processed$Type_of_Cab=as.numeric(train_processed$Type_of_Cab)

preProcValues <- preProcess(train[,2:13], method = c("medianImpute","center","scale"))
train_processed[,2:13] <- predict(preProcValues, train_processed[,2:13])
#My_train=predict(preProcValues,train[,2:13])


train_processed$Var=train_processed$Var1*train_processed$Var2*train_processed$Var3
#Backword Rejection method
#-----------------------------------------------------------------------------------------------




#-----------------------------------------------------------------------------------------------
# BACKWARD REJECTION METHOD TO SELECT BEST FEATURES
#-----------------------------------------------------------------------------------------------
predictors<-c('Type_of_Cab','Trip_Distance','Customer_Rating','Life_Style_Index','Destination_Type','Confidence_Life_Style_Index') RF:0.6301953 GBM:0.6622951
predictors<-c('Type_of_Cab','Trip_Distance','Customer_Rating','Life_Style_Index','Destination_Type')  RF:0.6293182  RF:0.6315360  GBM:0.6629681   XGB:0.6705056
predictors<-c('Type_of_Cab','Trip_Distance','Customer_Rating','Life_Style_Index','Destination_Type','Var')  rf:0.6110809
predictors<-c('Type_of_Cab','Trip_Distance','Customer_Rating','Life_Style_Index','Destination_Type','Cancellation_Last_1Month')  RF:0.6168476t
predictors<-c('Type_of_Cab','Trip_Distance','Customer_Rating','Life_Style_Index')  RF: 0.6253210 ,0.6206451
predictors<-c('Type_of_Cab','Trip_Distance','Customer_Rating','Destination_Type')  RF:0.6219771  
predictors<-c('Type_of_Cab','Trip_Distance','Life_Style_Index','Destination_Type') RF:0.6064260
predictors<-c('Type_of_Cab','Trip_Distance','Customer_Rating','Life_Style_Index')  rf:0.6220061
predictors<-c('Type_of_Cab','Trip_Distance','Customer_Rating') rf:0.6183628
predictors<-c('Type_of_Cab','Trip_Distance') RF:0.5747445
predictors<-c('Type_of_Cab')


outcomeName<-c('Surge_Pricing_Type')
  




#--------------------------------------------------------------------------------------------------------
##Training base models using Caret(without parm Tuning)
#--------------------------------------------------------------------------------------------------------
names(getModelInfo())   # to retrive all th emodels in CARET
#We can simply apply a large number of algorithms with similar syntax
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm')
#--------------------------------------------------------------------------------------------------------



fitControl <- trainControl(
  method = "repeatedcv", # Repeated Cross validation   
  number = 5,            # Five fold cross validation
  repeats = 1)           # Repeat five times

#-----------------------------------------------------------------------------------------------------------------------#

modelLookup(model='gbm')

gbmGrid <- expand.grid(n.trees=c(10,20,50,100,500,1000),
                    shrinkage=c(0.01,0.05,0.1,0.5),
                    n.minobsinnode = c(3,5,10),
                    interaction.depth=c(1,5,10))


model_gbm<-train(train_processed[,predictors],
                 train_processed[,outcomeName],
                 method='gbm',
                 trControl=fitControl,
                 tuneGrig=gbmGrid)




#Type_of_Cab                 100.0000    -1
#Trip_Distance                 4.1639    -1
#Cancellation_Last_1Month      4.0141
#Customer_Rating               3.3433    -1
#Life_Style_Index              0.3999    -1
#Confidence_Life_Style_Index   0.0000
#Destination_Type              0.0000
#Customer_Since_Months         0.0000
#Var2                          0.0000
#Var3                          0.0000
#Gender                        0.0000
#Var1                          0.0000

#--------------------------------------------------------------------------------------------------------------------#
modelLookup(model='xgbTree')

xgbGrid <- expand.grid(
  eta = 0.3,
  max_depth = 1,
  nrounds = 400,
  gamma = 0,                  #default=0
  colsample_bytree=.8,
  min_child_weight=1,
  subsample=1)


model_xgbTree<-train(x=train_processed[,predictors],
                     y=train_processed[,outcomeName],
                     method="xgbTree"
                     objective = "multi:softprob")
                     metric = "Accuracy",
                     num_class=3,
                     tuneGrig='xgbGrid',
                     trControl='fitControl'
                     )


#Type_of_Cab                 100.00000
#Cancellation_Last_1Month      9.33191
#Trip_Distance                 7.95146
#Customer_Rating               6.69269
#Var1                          4.55456
#Life_Style_Index              4.05084
#Confidence_Life_Style_Index   3.66227
#Destination_Type              3.09305
#Var3                          2.78014
#Customer_Since_Months         0.03636
#Gender                        0.00000

#--------------------------------------------------------------------------------------------------------------------#

modelLookup(model='rf')

model_rf<-train(train_processed[,predictors],
                train_processed[,outcomeName],
                 method='rf')


model_rf<-train(data.matrix(train_processed[,predictors]),
                data.matrix(train_processed[,outcomeName]),
                method='rf',
                trControl=fitControl,
                tuneLength=10)

#Type_of_Cab                 100.000
#Trip_Distance                46.693
#Customer_Rating              44.572
#Var3                         38.963
#Life_Style_Index             34.656
#Var2                         23.496
#Var1                         21.160
#Customer_Since_Months        18.747
#Cancellation_Last_1Month     10.084
#Destination_Type              8.928
#Confidence_Life_Style_Index   5.217
#Gender                        0.000


#-------------------------------------------------------------------------------------------------------------
# Other Functions
#-------------------------------------------------------------------------------------------------------------

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

nrow(gbmGrid)
modelLookup(model='gbm')  #To get the tuneing parms of a model

# summarizing the model
print(model_gbm)
max(model_gbm$results$Accuracy)
plot(model_gbm)     #with the two dimwntional plots we can realize the parameters desired
plot(varImp(object=model_rf),main="RF - Variable Importance")
names(getModelInfo())  # TO get all the model informations
#-------------------------------------------------------------------------------------------------------------

#Overall
#Type_of_Cab                 100.000
#Trip_Distance                24.386
#Customer_Rating              23.291
#Life_Style_Index             21.785
#Var1                         19.310
#Var3                         15.842
#Var2                         11.139
#Customer_Since_Months         9.286
#Confidence_Life_Style_Index   7.728
#Cancellation_Last_1Month      4.756
#Destination_Type              3.836
#Gender                        0.000



#-----------------------------------------------------------------------------------------------
# C O N C L U S I O N         A N D          O B SER V AT I O N 
#-----------------------------------------------------------------------------------------------
#By taking just 100 samples from each category and build a XGB model, the model has the capability of having 0.6423418 on the entire training test.
#By taking just 100 samples from each category and build a XGB model, the model has the capability of having 0.64222126 on the entire training test.
