#-----------------------------------------------------------------------
#Strategy 
##Added extra features and imputed missing data
##found most significant features using wrapper package (Boruta)
##Found most suitable algorithm using CARET model and tuned manually
##Run XGB model withthe most IMP features
#-----------------------------------------------------------------------

#--------------------------------------------------
# XGBOOST implementation
#--------------------------------------------------
#Reference
#https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/
#https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
#https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html


library(xgboost)
library(Matrix)
require(Matrix)
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




#-----------------------------------------------------------------------------------------
# imputation follows
#-----------------------------------------------------------------------------------------
train_cols=c( 
  "Item_Identifier",
  "Item_Weight",
  "Item_Fat_Content",
  "Item_Visibility",
  "Item_Type",
  "Item_MRP",
  "Outlet_Identifier",
  "Outlet_Establishment_Year",
  "Outlet_Size",              
  "Outlet_Location_Type",
  "Outlet_Type",
  #   "Item_Outlet_Sales",   #Target value
  "prod_code_text",
  "prod_code_num",  
  "outlet_age"    )

look_up=rbind(train_nona[train_cols],test_nona[train_cols])   #build a lookup table with only available data

#------------------------------------------------------------------------------------
# Get the missing Item weights of products from other stores for boths test  and train datasets
#------------------------------------------------------------------------------------
for (i in 1:nrow(train)) {
  if (is.na(train$Item_Weight[i])=='TRUE'){
    item<-as.character(train$Item_Identifier[i])
    temp_df=subset(look_up, Item_Identifier==item)
    train$Item_Weight[i]<-mean(temp_df$Item_Weight[1])
  }  
}


for (i in 1:nrow(test)) {
  if (is.na(test$Item_Weight[i])=='TRUE'){
    item<-as.character(test$Item_Identifier[i])
    temp_df=subset(look_up, Item_Identifier==item)
    test$Item_Weight[i]<-mean(temp_df$Item_Weight)
  }  
}

rm(test_nona,train_nona,temp_df,look_up) # now Delete un necessary DFs created




#------------------------------------------------------------------------------------------
# We got these most IMP features from Boruta package
#-------------------------------------------------------------------------------------------
set.seed=123
fea_col=
  c(
    "Item_Visibility",   #Removed since not unimportant in most of the models (observed in CARET)
    "Item_MRP",
    "Outlet_Identifier",
    "Outlet_Establishment_Year",
    "Outlet_Location_Type",
    "Outlet_Type",
    "prod_code_num"      #Removed since not unimportant in most of the models (observed in CARET)
  )
#----------------------------------------------
X=train[fea_col]
sparse_matrix<-sparse.model.matrix(  ~ .-1, data=X)
#--------Test Data------------------------------
X_test=test[fea_col]
sparse_matrix_test<-sparse.model.matrix(  ~ .-1, data=X_test)
#-----------------------------------------------

#---------------------------------------------------------------------------------------------------------
#Target Vector
#---------------------------------------------------------------------------------------------------------
output_vector <- train[,"Item_Outlet_Sales"]



#---------------------------------------------------------------
# Building Model: GBM
#---------------------------------------------------------------
XGB <- xgboost(data = sparse_matrix, 
               label = output_vector, 
               eta = 0.003,                      # Learning rate
               max_depth = 4,    
               min_child_weight=4,  
               colsample_bytree=.5,  #ratio of columns when constructing tree Default is :1
               subsample = 0.60,
               set.seed=2126,
               objective = "reg:linear",  #Linear Regression
               nthread = 4,
               nround=5600 
               
)

#[5599]	train-rmse:988.147440
#LB Score:1135.657883

save(XGB,file="BMS_XGB_final_model.Rdata")  
load("BMS_XGB_final_model.Rdata")

#--------------------------------------------------------------------------------------------------------
# Submitting LB score
#--------------------------------------------------------------------------------------------------------
y_pred<- predict(XGB,sparse_matrix_test)
test_result =as.data.frame(test[c("Item_Identifier","Outlet_Identifier")])
y_pred<- as.data.frame(y_pred)
test_result['Item_Outlet_Sales'] = y_pred
Sales_file=test_result

#If there are any negative shopping figures thats should be replaces with zero.-ve has no meaning
library(dplyr)
filter(Sales_file,Item_Outlet_Sales<0) # These negative shopping figures should be replaced with zero
Sales_file[c(which(Sales_file$Item_Outlet_Sales<0)),]$Item_Outlet_Sales=0
# The below is the final output loaded to AV site
write.csv(Sales_file,file='Sales_Result_GBM.csv',row.names=FALSE)



