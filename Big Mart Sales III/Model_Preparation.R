#--------------------------------------------------
# XGBOOST implementation
#--------------------------------------------------
#Referance
#https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/
#https://cran.r-project.org/web/packages/xgboost/xgboost.pdf
#https://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html


library(xgboost)
require(Matrix)


#----------------------------
# Use most important features
#----------------------------
fea_imp=c( "Item_Weight" ,   "Item_Fat_Content",         
           "Item_Visibility",       "Item_Type" ,     "Item_MRP"  ,          
           "Outlet_Identifier",     "outlet_age",     "Outlet_Size",            
           "Outlet_Location_Type",  "Outlet_Type"   )

X=train[fea_imp]
sparse_matrix<-sparse.model.matrix(  ~ .-1, data=X)

#--------Test Data------------------------------
X_test=test[fea_imp]
sparse_matrix_test<-sparse.model.matrix(  ~ .-1, data=X_test)


#---------------------------------------
temp_train=as.matrix(sparse_matrix)
temp_train=as.data.frame(temp_train)

temp_test=as.matrix(sparse_matrix_test)
temp_test=as.data.frame(temp_test)

temp=rbind(temp_test,temp_train)
#---Only Desired featured for building tree------#
cols=c("Item_MRP", "Outlet_TypeSupermarket Type1", "Outlet_IdentifierOUT027",
       "outlet_age","Item_Visibility","Outlet_TypeSupermarket Type3",
       "Item_Weight","Outlet_IdentifierOUT019","Outlet_SizeSmall",
       "Outlet_IdentifierOUT018","Outlet_Location_TypeTier 3","Outlet_SizeMedium")
temp1=temp
temp1[cols]<-NULL   #Remove the high important columns

#Addinga Clustering feature
KMC = kmeans(temp1, centers = 10, iter.max = 1000)
cluster_mean=aggregate(temp1,by=list(KMC$cluster),FUN=mean)   # cluster mean
table(KMC$cluster)    #Each cluster size
rm(temp1)
temp1=temp
temp["cluster"]=as.data.frame(KMC$cluster)  # add clusters


#------------------------------------
#--Split the test and train data from the clustered combined(test train) data set.
#------------------------------------
what_toget=cols
what_toget=append(what_toget,"cluster")
what_tolook=cols
train_data=lookup_table(temp_train,temp,what_tolook,what_toget)
test_data=lookup_table(temp_test,temp,what_tolook,what_toget)
#traisform as matrix
X=as.matrix(test_data)
sparse_matrix_test=X


#-------------------------------------
# Get corresponding lables
#-------------------------------------
fea_col1=append(fea_imp,"Item_Outlet_Sales")
X=train[fea_col1]
sprs_mat<-sparse.model.matrix(  ~ .-1, data=X)
train_mat=as.matrix(sprs_mat)
train_mat=as.data.frame(train_mat)
what_toget1=append(what_toget,"Item_Outlet_Sales")
lable_train_data=lookup_table(train_data,train_mat,what_tolook,what_toget1)
sparse_matrix=lable_train_data[what_toget]
sparse_matrix=as.matrix(sparse_matrix)
output_vector <- lable_train_data[,"Item_Outlet_Sales"]







#----------------------------------------

#Target Vector
#output_vector <- train_nona[,"Item_Outlet_Sales"]
output_vector <- train[,"Item_Outlet_Sales"]


#----------------------------
# Tune and run the model
#----------------------------
#xgb <- xgboost(data = data.matrix(X[,-1]), 
GBM <- xgboost(data = sparse_matrix, 
               label = output_vector, 
               eta = 0.003,                      # Learning rate
               #max_depth = 4,   #.75
               #max_depth = 3,   #.756
               #max_depth = 2,   #.7569
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
               nround=3500 
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

#Feature importance
importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]],model=GBM)
head(importance)

#----------------------------------------------------
#Feature        Gain      Cover Frequence
#1:                     Item_MRP 0.49561820 0.113929919     0.114
#2:      Outlet_IdentifierOUT027 0.13068776 0.028022833     0.028
#3:      Outlet_IdentifierOUT019 0.07353293 0.007036211     0.007
#4: Outlet_TypeSupermarket Type1 0.05980857 0.039973150     0.040
#5:    Outlet_Establishment_Year 0.03521593 0.082157568     0.082
#6: Outlet_TypeSupermarket Type3 0.03079410 0.010016116     0.010
#----------------------------------------------------

#Plotting Feature importance
xgb.plot.importance(importance_matrix = importanceRaw)
xgb.plot.importance(importance_matrix)


#--------------------------------------------------------------------
#Testining model
#--------------------------------------------------------------------
test <- read.csv("C:/Users/Admin/Downloads/Big Mart Sales III/Test_u94Q5KV.csv", na.strings=c("","NA"))
test_nona=test[complete.cases(test),]   #Remove NA data
test['prod_code']=substr(test$Item_Identifier,start=1,stop=3)

#cols=c("Item_Identifier",    "Item_Weight"               , "Item_Fat_Content",         
#       "Item_Visibility",    "Item_Type"                 , "Item_MRP"        ,          
#       "Outlet_Identifier",  "Outlet_Establishment_Year" , "Outlet_Size"     ,            
#       "Outlet_Location_Type",      "Outlet_Type"            )


#X_test=test_nona[cols]
X_test=test[cols_init]
sparse_matrix_test<-sparse.model.matrix( ~ .-1, data=X_test)


#---------------------------------------
temp_test=as.matrix(sparse_matrix_test)
temp_test=as.data.frame(temp_test)
cols=c("Item_MRP", "Outlet_TypeSupermarket Type1", "Outlet_IdentifierOUT027",
       "outlet_age","Item_Visibility","Outlet_TypeSupermarket Type3",
       "Item_Weight","Outlet_IdentifierOUT019","Outlet_SizeSmall",
       "Outlet_IdentifierOUT018","Outlet_Location_TypeTier 3","Outlet_SizeMedium")
temp1=temp
temp1[cols]<-NULL   #Remove the high important columns

#Addinga Clustering feature
KMC = kmeans(temp1, centers = 10, iter.max = 1000)
cluster_mean=aggregate(temp1,by=list(KMC$cluster),FUN=mean)   # cluster mean
table(KMC$cluster)    #Each cluster size
temp=temp[,cols]
temp["cluster"]=as.data.frame(KMC$cluster)  # add clusters


#traisform as matrix
X=as.matrix(temp)
#sparse_matrix<-sparse.model.matrix(  ~ .-1, data=X)
sparse_matrix_test=X
#----------------------------------------
#----------------------------------------

y_pred<- predict(GBM,sparse_matrix_test)

test_result =as.data.frame(test[c("Item_Identifier","Outlet_Identifier")])
y_pred<- as.data.frame(y_pred)
test_result['Item_Outlet_Sales'] = y_pred
Sales_file=test_result
row.names(Sales_file)<-NULL
#Loan_file$Loan_Status[Loan_file$Loan_Status>=.5] <- "Y"
#Loan_file$Loan_Status[Loan_file$Loan_Status<.5] <- "N"

write.csv(Sales_file,file='Sales_Result_GBM.csv')


#Once the missing Credit_history feature has been categorized as 3(adifferent from existing )
#Test score-->0.188925
#LB Score --->0.7777778


