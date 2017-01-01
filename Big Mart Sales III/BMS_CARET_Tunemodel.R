#--------------------------------------------------------------------------------
#CARET implementation in order to choose best model and best hyper parameters 
# Documentarion on CARET
#http://topepo.github.io/caret/train-models-by-tag.html
#--------------------------------------------------------------------------------
install.packages("ggplot2")       #Dependents for CARET
install.packages("scales")        #Dependents for CARET
install.packages("ModelMetrics")  #Dependents for CARET
install.packages("caret", dependencies = c("Depends", "Suggests"))
library("ModelMetrics")           #Dependents for CARET
library("ggplot2")                #Dependents for CARET 





#------------------------------------------------------------------------------------------
#Below is the preprocessing of train data
#------------------------------------------------------------------------------------------
train <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Train_UWu5bXk.csv", na.strings=c("","NA","NaN"))
# Imputation (a case where missing value depends on another IV)
# by exploration it is found that all Small Outlet_sizes are missing 
train$Outlet_Size[is.na(train$Outlet_Size)]<-'Small'  # imputation (Missing depends on missing. Identified by exploration on Excel)
train$Item_Fat_Content=as.character(train$Item_Fat_Content)
train[c(row.names(train[ which(train$Item_Type=='Household' | train$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
train[c(which(train$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
train[c(which(train$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
train$Item_Fat_Content=as.factor(train$Item_Fat_Content)
#Feature extraction 
train['prod_code_text']=substr(train$Item_Identifier,start=1,stop=3)
train['prod_code_num']=substr(train$Item_Identifier,start=4,stop=6)
train['outlet_age']=2016-train$Outlet_Establishment_Year
train_nona=train[complete.cases(train),]   #Remove NA data




#------------------------------------------------------------------------------------------
#Below is the preprocessing of test data
#------------------------------------------------------------------------------------------
test <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Test_u94Q5KV.csv", na.strings=c("","NA"))
# Imputation (a case where missing value depends on another IV)
# by exploration it is found that all Small Outlet_sizes are missing 
test$Outlet_Size[is.na(test$Outlet_Size)]<-'Small'  
# Cleaning activity
test$Item_Fat_Content=as.character(test$Item_Fat_Content)
test[c(row.names(test[ which(test$Item_Type=='Household' | test$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
test[c(which(test$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
test[c(which(test$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
test$Item_Fat_Content=as.factor(test$Item_Fat_Content)
#Feature extraction 
test['prod_code_text']=substr(test$Item_Identifier,start=1,stop=3)
test['prod_code_num']=substr(test$Item_Identifier,start=4,stop=6)
test['outlet_age']=2016-test$Outlet_Establishment_Year
test_nona=test[complete.cases(test),]   #Remove NA data




#Loading caret package
library("caret")




fea_col=
  c("Item_Visibility",
    "Item_MRP",
    "Outlet_Identifier",
    "Outlet_Establishment_Year",
    "Outlet_Location_Type",
    "Outlet_Type",
    "prod_code_num")




#-------------------------------------------------------------------------------------
# only relevant features given by Boruta are chosen to train
#-------------------------------------------------------------------------------------
train_master=train[,fea_col]
train_master$Item_Outlet_Sales=train[,"Item_Outlet_Sales"]
  



#-------------------------------------------------------------------------------------
# Ramdomsample Master data for train data set
#-------------------------------------------------------------------------------------
index <- createDataPartition(train_master$Item_Outlet_Sales, p=0.05, list=FALSE)
train_processed <- train_master[ index,]
testSet <- train_master[-index,]




#-------------------------------------------------------------------------------------
# Train different models and choose one based on the RMSE 
#-------------------------------------------------------------------------------------
require(Matrix)
BMS_glm<-train(Item_Outlet_Sales ~ . +Item_Visibility:prod_code_num,
                   data=train_processed,
                   method='glm')
load("BMS_glm.Rdata")
save(BMS_glm,file="BMS_glm.Rdata")
plot(varImp(object=BMS_glm))
#-------------------------------------------------------------------------------------
#RMSE      Rsquared 
#1136.881  0.5577641
#-------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------
BMS_xgbTree<-train(Item_Outlet_Sales ~ . +Item_Visibility:prod_code_num,
                     data=train_processed,
                     method='xgbTree')
load("BMS_xgbTree.Rdata")
save(BMS_xgbTree,file="BMS_xgbTree.Rdata")
plot(varImp(object=BMS_xgbTree))
BMS_xgbTree$results
#52  0.3         3     0              0.8                1      1.00      50 1106.091 0.5842621 20.40753
BMS_xgbTree$bestTune
#        nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
#52      50         3 0.3     0              0.8                1         1
#-------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------
BMS_rf<-train(Item_Outlet_Sales ~ . +Item_Visibility:prod_code_num,
                   data=train_processed,
                   method='rf')
load("BMS_rf.Rdata")
save(BMS_rf,file="BMS_rf.Rdata")
plot(varImp(object=BMS_rf))
#BMS_rf$results
#mtry  RMSE      Rsquared 
#2   1546.966  0.2469320
#68   1190.418  0.5089131
#135   1206.054  0.4958701

#BMS_rf$bestTune
#mtry
#2   68

#-------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------
BMS_bgm<-train(Item_Outlet_Sales ~ . +Item_Visibility:prod_code_num,
                 data=train_processed,
                 method='gbm')
load("BMS_bgm.Rdata")
save(BMS_bgm,file="BMS_bgm.Rdata")
plot(varImp(object=BMS_bgm))
#BMS_bgm$results
#shrinkage interaction.depth n.minobsinnode n.trees     RMSE  Rsquared   RMSESD
#1       0.1                 1             10      50 1238.687 0.4979157 22.76582
#4       0.1                 2             10      50 1126.886 0.5700114 19.87491
#7       0.1                 3             10      50 1096.241 0.5877971 19.26651
#2       0.1                 1             10     100 1177.915 0.5266455 19.72352
#5       0.1                 2             10     100 1101.772 0.5818632 18.95213
#8       0.1                 3             10     100 1087.610 0.5912848 19.28603
#3       0.1                 1             10     150 1157.813 0.5392885 18.70393
#6       0.1                 2             10     150 1097.751 0.5837961 19.08201
#9       0.1                 3             10     150 1087.806 0.5906230 19.43140
#------------------
#BMS_bgm$bestTune
##   n.trees interaction.depth shrinkage n.minobsinnode
#8     100                 3       0.1             10
#-------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------
BMS_nnet<-train(Item_Outlet_Sales ~ . +Item_Visibility:prod_code_num,
                 data=train_processed,
                 method='nnet')
load("BMS_nnet.Rdata")
save(BMS_nnet,file="BMS_nnet.Rdata")
plot(varImp(object=BMS_nnet))
BMS_nnet$results
#---------------------------------
#size decay     RMSE Rsquared   RMSESD RsquaredSD
#1    1 0e+00 2773.567      NaN 29.39298         NA
#----------------------------------
#-------------------------------------------------------------------------------------
Conclusion : both GBM and XGB were found good for implementation.
But I prefered to XGB since regularization is possible 
