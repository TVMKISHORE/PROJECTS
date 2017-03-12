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
#load test and train data sets
#----------------------------
train <- read.csv("C:/Users/Admin/Downloads/Loan Prediction III/train_u6lujuX_CVtuZ9i.csv", na.strings=c("","NA"))
train_nona=train[complete.cases(train),]   #Remove NA data

#cols=c("Credit_History","ApplicantIncome","LoanAmount","CoapplicantIncome","Loan_Amount_Term")
cols=c("Credit_History","ApplicantIncome","LoanAmount","CoapplicantIncome")
X=train_nona[cols]
#train_nona$Loan_Status<-recode(train_nona$Loan_Status,"'Y'=1;'N'=0")     #  To recode the target var
#sparse.model.matrix(~.-1,data=df)                                        #  When you need all columns
#output_vector<-sparse.model.matrix( ~ .-1, data=X["Loan_Status"])        #  To exclude columns
sparse_matrix<-sparse.model.matrix(  ~ .-1, data=X)

#Target Vector
output_vector <- train_nona[,'Loan_Status']=="Y"


#----------------------------
# Tune and run the model
#----------------------------
#xgb <- xgboost(data = data.matrix(X[,-1]), 
xgb <- xgboost(data = sparse_matrix, 
               label = output_vector, 
               eta = 0.01,                      # Learning rate
               #max_depth = 4,   #.75
               #max_depth = 3,   #.756
               #max_depth = 2,   #.7569
               max_depth = 1,    #.76388
               min_child_weight=2,  #
               colsample_bytree=.5,  #Tatio of columns when constructing tree Default is :1
               subsample = 0.60,
               set.seed=13,
               objective = "binary:logistic",   # logistic regression for binary classification. Output probability.
               #objective = "binary:logitraw",  #logistic regression for binary classification, outputscore before logistic transformation
               nthread = 4,
               #lambda=.1,     #L2 regularization
               #alpha=.100     #L1 regularization
               #num_parallel_tree=100,
               nround=1000 
               #early.stop.round=10  # same result observed in 'n' rounds, then stops execution
)


#Feature importance
importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]],model=xgb)
head(importance)

#----------------------------------------------------
#Top features based on the importance

#Feature       Gain      Cover Frequence
#1:    Credit_History 0.65699111 0.28693141     0.273
#2:        LoanAmount 0.13214564 0.24986223     0.254
#3:   ApplicantIncome 0.10070495 0.21589012     0.223
#4: CoapplicantIncome 0.09737886 0.21094972     0.212
#5:  Loan_Amount_Term 0.01277944 0.03636651     0.038
#----------------------------------------------------

#Plotting Feature importance
xgb.plot.importance(importance_matrix = importanceRaw)
xgb.plot.importance(importance_matrix)



#-------------------------------
# Model evaluation 
#-------------------------------
y_pred <- predict(xgb, sparse_matrix)
y_pred<- as.data.frame(y_pred)


train_result =train_nona["Loan_ID"]
train_result['model_pred'] = y_pred
train_result['Loan_status'] = train_nona["Loan_Status"]
tapply(train_result$model_pred,train_result$Loan_status=='Y',max)
temp1=subset(train_result,model_pred >=0.5 & Loan_status=='Y')
temp2=subset(train_result,model_pred <0.5 & Loan_status=='N')


temp1=subset(train_result,model_pred >=e & Loan_status=='Y')
temp2=subset(train_result,model_pred <e & Loan_status=='N')

bad1=subset(train_result,model_pred >=e & Loan_status=='N')
bad2=subset(train_result,model_pred <e & Loan_status=='Y')

temp1=temp1[order(temp1$model_pred),]
bad1=bad1[order(bad1$model_pred),]

temp2=temp1[order(temp1$model_pred),]
bad2=bad1[order(bad1$model_pred),]


write.csv(train_result,file='Result_GBM.csv')




#--------------------------------------------------------------------
#Testining model
#--------------------------------------------------------------------
cols=c("Credit_History.y","ApplicantIncome.y","LoanAmount.y","CoapplicantIncome.y","Loan_Amount_Term.y")
X_test=test[cols]

sparse_matrix_test<-sparse.model.matrix( ~ .-1, data=X_test)

y_pred<- predict(xgb,sparse_matrix_test)

test_result =as.data.frame(test["Loan_ID"])
y_pred<- as.data.frame(y_pred)
test_result['Loan_Status'] = y_pred
Loan_file=test_result
Loan_file$Loan_Status[Loan_file$Loan_Status>=.5] <- "Y"
Loan_file$Loan_Status[Loan_file$Loan_Status<.5] <- "N"

write.csv(Loan_file,file='Result_GBM.csv')


#Once the missing Credit_history feature has been categorized as 3(adifferent from existing )
#Test score-->0.188925
#LB Score --->0.7777778


