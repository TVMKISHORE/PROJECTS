#------------------------------------------------------------------------------------------
##Strategy 
##Added extra features 
##Applied XGB algorithm
##From the algorithm , we got the most IMP features
##Take the most IMP 12 features and exclude all to reduce noise.(the selection is done after parse) and create model
##Clustered non-IMP features and used cluster ID as feature , but no major improvement.
##XGB model has been tuned well
#------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------
#Below is the preprocessing of train data
#------------------------------------------------------------------------------------------
train <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Train_UWu5bXk.csv", na.strings=c("","NA","NaN"))
train$Outlet_Size[is.na(train$Outlet_Size)]<-'Small'  # imputation
train$Item_Fat_Content=as.character(train$Item_Fat_Content)
train[c(row.names(train[ which(train$Item_Type=='Household' | train$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
train[c(which(train$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
train[c(which(train$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
train$Item_Fat_Content=as.factor(train$Item_Fat_Content)
train['prod_code']=substr(train$Item_Identifier,start=1,stop=3)
train['outlet_age']=2016-train$Outlet_Establishment_Year
train_nona=train[complete.cases(train),]   #Remove NA data



#------------------------------------------------------------------------------------------
#Below is the preprocessing of test data
#------------------------------------------------------------------------------------------
test <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Test_u94Q5KV.csv", na.strings=c("","NA"))
test$Outlet_Size[is.na(test$Outlet_Size)]<-'Small'  # imputation
test['prod_code']=substr(test$Item_Identifier,start=1,stop=3)
test$Item_Fat_Content=as.character(test$Item_Fat_Content)
test[c(row.names(test[ which(test$Item_Type=='Household' | test$Item_Type == 'Health and Hygiene'), ])),]$Item_Fat_Content="Non-Con"
test[c(which(test$Item_Fat_Content=='Low Fat')),]$Item_Fat_Content='LF'
test[c(which(test$Item_Fat_Content=='Regular')),]$Item_Fat_Content='Reg'
test$Item_Fat_Content=as.factor(test$Item_Fat_Content)
test['prod_code']=substr(test$Item_Identifier,start=1,stop=3)
test['outlet_age']=2016-test$Outlet_Establishment_Year
test_nona=test[complete.cases(test),]   #Remove NA data




#-----------------------------------------------------------------------------------------
# Imputation follows
#-----------------------------------------------------------------------------------------
train_cols=c(  "Item_Identifier"           ,"Item_Weight"               ,"Item_Fat_Content"         
               ,"Item_Visibility"           ,"Item_Type"                 ,"Item_MRP"                 
               ,"Outlet_Identifier"         ,"Outlet_Establishment_Year" ,"Outlet_Size"              
#              ,"Outlet_Location_Type"      ,"Outlet_Type"               ,"Item_Outlet_Sales"        
               ,"Outlet_Location_Type"      ,"Outlet_Type"               ,"prod_code"         
               ,"outlet_age"      )

look_up=rbind(train_nona[train_cols],test_nona[train_cols])

#------------------------------------------------------------------------------------
# Get the missing Item weights from other stores. For boths test  and train datasets
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
# cluster data using non important features
#-------------------------------------------------------------------------------------------
set.seed=123
fea_col=c( "Item_Weight" ,   "Item_Fat_Content",         
           "Item_Visibility",       "Item_Type" ,     "Item_MRP"  ,          
           "Outlet_Identifier",     "outlet_age",     "Outlet_Size",            
           "Outlet_Location_Type",  "Outlet_Type"   )
#----------------------------------------------
X=train[fea_col]
sparse_matrix<-sparse.model.matrix(  ~ .-1, data=X)
#--------Test Data------------------------------
X_test=test[fea_col]
sparse_matrix_test<-sparse.model.matrix(  ~ .-1, data=X_test)
#-----------------------------------------------
fea_imp=c("Item_MRP", "Outlet_TypeSupermarket Type1", "Outlet_IdentifierOUT027",
       "outlet_age","Item_Visibility","Outlet_TypeSupermarket Type3",
        "Item_Weight","Outlet_IdentifierOUT019","Outlet_SizeSmall",
        "Outlet_IdentifierOUT018","Outlet_Location_TypeTier 3","Outlet_SizeMedium")
sparse_matrix=as.matrix(sparse_matrix)
sparse_matrix=as.data.frame(sparse_matrix)
temp=sparse_matrix
temp[fea_imp]<-NULL
KMC = kmeans(temp, centers = 10, iter.max = 2000)
sparse_matrix=sparse_matrix[,fea_imp]
sparse_matrix["cluster"]=as.data.frame(KMC$cluster)  # add clusters
sparse_matrix=as.matrix(sparse_matrix)

sparse_matrix_test=as.matrix(sparse_matrix_test)
sparse_matrix_test=as.data.frame(sparse_matrix_test)
temp=sparse_matrix_test
temp[fea_imp]<-NULL
KMC = kmeans(temp, centers = 10, iter.max = 2000)
sparse_matrix_test=sparse_matrix_test[,fea_imp]
sparse_matrix_test["cluster"]=as.data.frame(KMC$cluster)  # add clusters
sparse_matrix_test=as.matrix(sparse_matrix_test)


#---------------------------------------------------------------------------------------------------------
#Target Vector
#---------------------------------------------------------------------------------------------------------
output_vector <- train[,"Item_Outlet_Sales"]



#---------------------------------------------------------------
# Building Model
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
train-rmse:1015.014726
LB-RMSE:1154.98607

