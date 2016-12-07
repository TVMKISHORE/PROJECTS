#-----------------------------------------------------------------------
#Strategy 
##Added extra features and imputed missing data
##find the most significant features using wrapper feature (Boruta package)
##From Step3, we got the most IMP features
##Run XGB model withthe most IMP features
#-----------------------------------------------------------------------

#--------------------------------------------------
# XGBOOST implementation
#--------------------------------------------------
#Referance
#https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/
#https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
#https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html


library(xgboost)
require(Matrix)
#------------------------------------------------------------------------------------------
#Below is the preprocessing of train data
#------------------------------------------------------------------------------------------
train <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Train_UWu5bXk.csv", na.strings=c("","NA","NaN"))
train$Outlet_Size[is.na(train$Outlet_Size)]<-'Small'  # imputation (Missing depends on missing. Identified by exploration on Excel)
train$Item_Fat_Content=as.character(train$Item_Fat_Content)
train[c(row.names(train[ which(train$Item_Type=='Household' | train$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
train[c(which(train$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
train[c(which(train$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
train$Item_Fat_Content=as.factor(train$Item_Fat_Content)
train['prod_code_text']=substr(train$Item_Identifier,start=1,stop=3)
train['prod_code_num']=substr(train$Item_Identifier,start=4,stop=6)
train['outlet_age']=2016-train$Outlet_Establishment_Year
train_nona=train[complete.cases(train),]   #Remove NA data

#------------------------------------------------------------------------------------------
#Below is the preprocessing of test data
#------------------------------------------------------------------------------------------
test <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Test_u94Q5KV.csv", na.strings=c("","NA"))
test$Outlet_Size[is.na(test$Outlet_Size)]<-'Small'  # imputation
test$Item_Fat_Content=as.character(test$Item_Fat_Content)
test[c(row.names(test[ which(test$Item_Type=='Household' | test$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
test[c(which(test$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
test[c(which(test$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
test$Item_Fat_Content=as.factor(test$Item_Fat_Content)
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

look_up=rbind(train_nona[train_cols],test_nona[train_cols])

#------------------------------------------------------------------------------------
# Get the missing Item weights from other stores for boths test  and train datasets
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

rm(test_nona,train_nona,temp_df,look_up) # Delete un necessary cols




#------------------------------------------------------------------------------------------
# We got these features from Boruta package
#-------------------------------------------------------------------------------------------
set.seed=123
fea_col=
c("Item_Visibility",
"Item_MRP",
"Outlet_Identifier",
"Outlet_Establishment_Year",
"Outlet_Location_Type",
"Outlet_Type",
"prod_code_num")
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
# Building & Tuning Model
#---------------------------------------------------------------
GBM <- xgboost(data = sparse_matrix, 
               label = output_vector, 
               eta = 0.003,                      # Learning rate
               max_depth = 4,    #.76388
               min_child_weight=4,  #
               colsample_bytree=.5,  #Tatio of columns when constructing tree Default is :1
               subsample = 0.60,
               set.seed=123,
               #objective = "binary:logistic",   # logistic regression for binary classification. Output probability.
               #objective = "binary:logitraw",  #logistic regression for binary classification, outputscore before logistic transformation
               #objective = "binary:logistic",   # logistic regression for binary classification. Output probability.
               objective = "reg:linear",  #Linear Regression
               nthread = 4,
               #lambda=.1,     #L2 regularization
               #alpha=.1000     #L1 regularization
               #num_parallel_tree=100,
               nround=3750 
               #early.stop.round=10  # same result observed in 'n' rounds, then stops execution
)

#--------------train-rmse:1175.734741--Nround=1000
#--------------train-rmse:1138.642678--Nround=2000
#--------------train-rmse:1127.480731--Nround=3000  Even after changing the OEY to a numeric 
#--------------train-rmse:1052.109323--Nround=3000  max_depth=2  LB:1173.97829
#--------------train-rmse:993.597484---Nround=3000  max_depth=3  LB:1173.97829
#--------------train-rmse:894.658724---Nround=4000  max_depth=4  LB:1176.59973(Getting over fit)
#--------------train-rmse:911.582145---Nround=3500  max_depth=4  LB:1172.3686(Score Improved)
#--------------train-rmse:910.801133---Nround=3500  max_depth=4  LB:1163.660414(Score Improved)




#--------------------------------------------------------------------------------------------------------
# Submitting LB score
#--------------------------------------------------------------------------------------------------------
y_pred<- predict(GBM,sparse_matrix_test)
test_result =as.data.frame(test[c("Item_Identifier","Outlet_Identifier")])
y_pred<- as.data.frame(y_pred)
test_result['Item_Outlet_Sales'] = y_pred
Sales_file=test_result
write.csv(Sales_file,file='Sales_Result_GBM.csv',row.names=FALSE)

#C:\Users\Admin\Documents
#After using features given by Boruta package
train-rmse:1024.771855
LB-RMSE:1150.064889
#After adding prod_code_num feature 
train-rmse:1013.952348
train-rmse:1137.235000


#Lookup function
lookup_table <- function(df1,df2,what_tolook,what_toget){
  temp_df = merge(df1,df2,by=what_tolook)
  ret_df  = temp_df[what_toget]
  return(ret_df)
}
