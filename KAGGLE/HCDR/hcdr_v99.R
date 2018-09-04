library(data.table)
#install.packages('sqldf')
library('sqldf')
library('caret')


test<-fread('application_test.csv')
train<-fread('application_train.csv')
length(setdiff(train$SK_ID_CURR,test$SK_ID_CURR))

#Sample to eliminate class imalance 
data_positive =subset(train,TARGET==1)
data_negative =subset(train,TARGET==0)
idx=sample(1:length(data_negative$SK_ID_CURR),24825,replace=FALSE)
data_neg_sam=data_negative[idx,]
my_train=rbind(data_neg_sam,data_positive)
rm(data_neg_sam,data_negative,data_positive,idx)
#--------------------------------------
# Encoding
keepnaforspace <- function (X){
  for (i in 1:length(X)){
  if (X[i]==""){
    X[i]=NA
  }
}
}

keepna_all_col <- function(my_train){
  keepnaforspace(my_train$NAME_CONTRACT_TYPE)
  keepnaforspace(my_train$CODE_GENDER)          
  keepnaforspace(my_train$FLAG_OWN_CAR)                
  keepnaforspace(my_train$FLAG_OWN_REALTY)
  keepnaforspace(my_train$NAME_TYPE_SUITE)
  keepnaforspace(my_train$NAME_INCOME_TYPE)
  keepnaforspace(my_train$NAME_EDUCATION_TYPE)
  keepnaforspace(my_train$NAME_FAMILY_STATUS)
  keepnaforspace(my_train$NAME_HOUSING_TYPE)
  keepnaforspace(my_train$OCCUPATION_TYPE)
  keepnaforspace(my_train$WEEKDAY_APPR_PROCESS_START)
  keepnaforspace(my_train$ORGANIZATION_TYPE )
  keepnaforspace(my_train$FONDKAPREMONT_MODE)
  keepnaforspace(my_train$HOUSETYPE_MODE )
  keepnaforspace(my_train$WALLSMATERIAL_MODE)
  keepnaforspace(my_train$EMERGENCYSTATE_MODE)
  
  my_train$NAME_CONTRACT_TYPE=as.factor(my_train$NAME_CONTRACT_TYPE)
  my_train$NAME_CONTRACT_TYPE=as.numeric(my_train$NAME_CONTRACT_TYPE)
  
  my_train$CODE_GENDER=as.factor(my_train$CODE_GENDER)
  my_train$CODE_GENDER=as.numeric(my_train$CODE_GENDER)
  
  my_train$FLAG_OWN_CAR=as.factor(my_train$FLAG_OWN_CAR)
  my_train$FLAG_OWN_CAR=as.numeric(my_train$FLAG_OWN_CAR)
  
  my_train$FLAG_OWN_REALTY=as.factor(my_train$FLAG_OWN_REALTY)
  my_train$FLAG_OWN_REALTY=as.numeric(my_train$FLAG_OWN_REALTY)
  
  my_train$NAME_TYPE_SUITE=as.factor(my_train$NAME_TYPE_SUITE)
  my_train$NAME_TYPE_SUITE=as.numeric(my_train$NAME_TYPE_SUITE)
  
  my_train$NAME_INCOME_TYPE=as.factor(my_train$NAME_INCOME_TYPE)
  my_train$NAME_INCOME_TYPE=as.numeric(my_train$NAME_INCOME_TYPE)
  
  my_train$NAME_EDUCATION_TYPE=as.factor(my_train$NAME_EDUCATION_TYPE)
  my_train$NAME_EDUCATION_TYPE=as.numeric(my_train$NAME_EDUCATION_TYPE)
  
  my_train$NAME_FAMILY_STATUS=as.factor(my_train$NAME_FAMILY_STATUS)
  my_train$NAME_FAMILY_STATUS=as.numeric(my_train$NAME_FAMILY_STATUS)
  
  my_train$NAME_HOUSING_TYPE=as.factor(my_train$NAME_HOUSING_TYPE)
  my_train$NAME_HOUSING_TYPE=as.numeric(my_train$NAME_HOUSING_TYPE)
  
  my_train$OCCUPATION_TYPE=as.factor(my_train$OCCUPATION_TYPE)
  my_train$OCCUPATION_TYPE=as.numeric(my_train$OCCUPATION_TYPE)
  
  my_train$WEEKDAY_APPR_PROCESS_START=as.factor(my_train$WEEKDAY_APPR_PROCESS_START)
  my_train$WEEKDAY_APPR_PROCESS_START=as.numeric(my_train$WEEKDAY_APPR_PROCESS_START)
  
  my_train$ORGANIZATION_TYPE=as.factor(my_train$ORGANIZATION_TYPE)
  my_train$ORGANIZATION_TYPE=as.numeric(my_train$ORGANIZATION_TYPE)
  
  my_train$FONDKAPREMONT_MODE=as.factor(my_train$FONDKAPREMONT_MODE)
  my_train$FONDKAPREMONT_MODE=as.numeric(my_train$FONDKAPREMONT_MODE)
  
  my_train$HOUSETYPE_MODE=as.factor(my_train$HOUSETYPE_MODE)
  my_train$HOUSETYPE_MODE=as.numeric(my_train$HOUSETYPE_MODE)
  
  my_train$WALLSMATERIAL_MODE=as.factor(my_train$WALLSMATERIAL_MODE)
  my_train$WALLSMATERIAL_MODE=as.numeric(my_train$WALLSMATERIAL_MODE)
  
  my_train$EMERGENCYSTATE_MODE=as.factor(my_train$EMERGENCYSTATE_MODE)
  my_train$EMERGENCYSTATE_MODE=as.numeric(my_train$EMERGENCYSTATE_MODE)
  #--------------------------------------
  # Imputation
  
  #--------------------------------------
  impute_model <- preProcess(my_train, method = c("knnImpute","center","scale"))
  impute_data<-predict(impute_model,my_train)
  return(impute_data)
  
}



train_data=keepna_all_col(my_train[,c(-1,-2)])
train_data$TARGET<- as.factor(my_train$TARGET)
#write.csv(train_data,'train_data.csv')  #train file saved

test_data=keepna_all_col(test[,c(-1)])
test_data$SK_ID_CURR<-test$SK_ID_CURR
write.csv(test_data,'test_data.csv')  #train file saved



#--------------------------------------
# Model building
#-------------------------------------


library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(train_data)
test.h2o <- as.h2o(test_data[,-(121)])
y.dep <- 121
x.indep <- c(1:120)


splits <- h2o.splitFrame(
  train.h2o,           ##  splitting the H2O frame we read above
  c(0.6,0.2),   ##  create splits of 60% and 20%; 
  ##  H2O will create one more split of 1-(sum of these parameters)
  ##  so we will get 0.6 / 0.2 / 1 - (0.6+0.2) = 0.6/0.2/0.2
  seed=1234)    ##  setting a seed will ensure reproducible results (not R's seed)



h2o.valid <- h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
h2o.train <- h2o.assign(splits[[3]], "test.hex")     ## R test, H2O test.hex


gbm <- h2o.gbm(         ## h2o.randomForest function
  training_frame = h2o.train,        ## the H2O frame for training
#  validation_frame = h2o.valid,      ## the H2O frame for validation (not required)
  x=x.indep,                        ## the predictor columns, by column index
  y=y.dep,                          ## the target index (what we are predicting)
  nfolds=10,
  seed = 1000000)                ## Set the random seed so that this can be
##  reproduced.

gbm@model$cross_validation_metrics_summary
h2o.auc(h2o.performance(gbm, newdata = h2o.valid))



finalgbm_predictions<-h2o.predict(
  object = gbm
  ,newdata = test.h2o)

Predictions=as.data.frame(finalgbm_predictions)
test_data$TARGET=Predictions$p1
write.table(test_data[,c(121,122)],'firstmodel.csv',row.names=FALSE,col.names=TRUE,sep=",")
