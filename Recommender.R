
# Practice Problem in analytics Vidya
# https://datahack.analyticsvidhya.com/contest/practice-problem-recommendation-engine/

train <- read.csv("C:/Users/kishore.thadiboyina/OneDrive/dataset/AV/Recommendation_system/train.csv")
test <- read.csv("C:/Users/kishore.thadiboyina/OneDrive/dataset/AV/Recommendation_system/test.csv")


#Following are the two separate aggregated data sets one group by user and the other by problem 
useragg <- read.csv("C:/Users/kishore.thadiboyina/OneDrive/dataset/AV/Recommendation_system/useragg.csv")
probagg <- read.csv("C:/Users/kishore.thadiboyina/OneDrive/dataset/AV/Recommendation_system/problemagg.csv")




names(probagg)=c("problem_id","prob.average.att","Max.prob.attempts","Min.prob.attempts" )
names(useragg)=c("user_id","Min.user.attempts","Max.user.attempts","user.average.att","Count.of.user" )






# -------------------------------------------------------------
#Users data imputation goes here.
users <- read.csv("C:/Users/v-vethad/OneDrive/dataset/AV/Recommendation_system/train_ps0mmDv_9yr6iGN/train/user_data.csv")
summary(users)
for (i in 1:length(users$country)){
  if (users$country[i]==""){
    users$country[i]=NA
  }
}
users$country_num=as.numeric(users$country)

join_users=merge(x=users,y=useragg,by='user_id',all.x=TRUE)
impute_model <- preProcess(join_users, method = c("knnImpute"))
library('RANN')
join_users_impute<-predict(impute_model,join_users)


# -------------------------------------------------------------
# Problem Data imputation 
problem <- read.csv("C:/Users/v-vethad/OneDrive/dataset/AV/Recommendation_system/train_ps0mmDv_9yr6iGN/train/problem_data.csv")
join_problem=merge(x=problem,y=probagg,by='problem_id',all.x=TRUE)

#Replace spaces with NA
for(i in 1:length(join_problem$level_type)){
  if (join_problem$level_type[i]==""){
    join_problem$level_type[i]=NA
  }
}

join_problem$level_type_num=as.numeric(join_problem$level_type_num)

for(i in 1:length(join_problem$tags)){
  if (join_problem$tags[i]==""){
    join_problem$tags[i]=NA
  }
}

join_problem$tags_num=as.numeric(join_problem$tags)

join_problem_complete=join_problem[complete.cases(join_problem),]
#join_problem_notcomplete=join_problem[!complete.cases(join_problem),]

impute_model1 <- preProcess(join_problem_complete, method = c("medianImpute"))
library('RANN')
join_problem_impute<-predict(impute_model2,join_problem)
#-----------------------------------------------------------------------



#-----------------------------------------------------------------------
#Joining features with the train set and test set
#https://stackoverflow.com/questions/1299871/how-to-join-merge-data-frames-inner-outer-left-right

#Blend with the user data set that and the problem dataset.
my_test_blend=merge(x=test,y=join_users_impute,by='user_id',all.x=TRUE)
my_train_blend=merge(x=train,y=join_users_impute,by='user_id',all.x=TRUE)

my_test_blend=merge(x=my_test_blend,y=join_problem_impute,by='problem_id',all.x=TRUE)
my_train_blend=merge(x=my_train_blend,y=join_problem_impute,by='problem_id',all.x=TRUE)

testSet  <- my_test_blend
trainSet <- my_train_blend
rm(my_test_blend,my_train_blend)

#modeling part

predictors <-names(trainSet[,c(-1,-2,-3,-7,-19,-21,-27)])
outcomeName <-names(trainSet)[27]
trainSet$target=as.factor(trainSet$attempts_range)

fitControl <- trainControl(
  method = "repeatedcv", # Repeated Cross validation   
  number = 3,            # Five fold cross validation
  repeats = 1)           # Repeat five times

train_data=trainSet[,c(predictors,outcomeName)]

model_knn<-train(target~.,data=train_data,
                 trControl=fitControl,method='knn')

testSet$predicted=predict(model_knn,testSet[,predictors])


temp=merge(test,testSet,by=c('user_id','problem_id'),all.x=TRUE)
upload<-testSet[,c("ID","predicted")]
names(upload)<-c("ID","attempts_range")
library("xlsx")
write.xlsx2(upload, file="C:/Users/v-vethad/OneDrive/dataset/AV/Recommendation_system/train_ps0mmDv_9yr6iGN/upload.xlsx", sheetName="sheet1", row.names=F)
