## loading libraries
install.packages("dummies")
library(dummies)
library(plyr)


train <- read.csv("C:/Users/Admin/Downloads/Black Friday/train_oSwQCTC/train.csv", stringsAsFactors=F)
test <- read.csv("C:/Users/Admin/Downloads/Black Friday/test_HujdGe7/test.csv", stringsAsFactors=F)


#How prices are distributed 
#The below graph shows that the prices are spread in a multy model distribution 
ggplot(train,aes(x=Purchase))+geom_histogram(binwidth = 1)
ggplot(train,aes(x=Purchase))+geom_density()
ggplot(train,aes(x=Purchase))+geom_dotplot(binwidth = 100)


#How prices and products distributed 
library(ggplot2)
#bivariable analysis 
scatterplot = ggplot(train, aes(x = Product_ID, y = Purchase))
scatterplot + geom_line()
scatterplot + geom_point()
#Observations: The product prices are not continues they can be categorized as buckets 


#How prices and users distributed 
scatterplot = ggplot(train, aes(x = User_ID, y = Purchase))
scatterplot + geom_point()

#Observations: The users prices are not continues they can be categorized as buckets 


X_train <- train
X_test <- test

# onehot-encoding city variable
X_train <- dummy.data.frame(X_train, names=c("City_Category"), sep="_")
X_test <- dummy.data.frame(X_test, names=c("City_Category"), sep="_")

# converting age variable to numeric
X_train$Age[X_train$Age == "0-17"] <- "15"
X_train$Age[X_train$Age == "18-25"] <- "21"
X_train$Age[X_train$Age == "26-35"] <- "30"
X_train$Age[X_train$Age == "36-45"] <- "40"
X_train$Age[X_train$Age == "46-50"] <- "48"
X_train$Age[X_train$Age == "51-55"] <- "53"
X_train$Age[X_train$Age == "55+"] <- "60"

X_test$Age[X_test$Age == "0-17"] <- "15"
X_test$Age[X_test$Age == "18-25"] <- "21"
X_test$Age[X_test$Age == "26-35"] <- "30"
X_test$Age[X_test$Age == "36-45"] <- "40"
X_test$Age[X_test$Age == "46-50"] <- "48"
X_test$Age[X_test$Age == "51-55"] <- "53"
X_test$Age[X_test$Age == "55+"] <- "60"

X_train$Age <- as.integer(X_train$Age)
X_test$Age <- as.integer(X_test$Age)

# converting stay in current city to numeric
X_train$Stay_In_Current_City_Years[X_train$Stay_In_Current_City_Years == "4+"] <- "4"
X_test$Stay_In_Current_City_Years[X_test$Stay_In_Current_City_Years == "4+"] <- "4"

X_train$Stay_In_Current_City_Years <- as.integer(X_train$Stay_In_Current_City_Years)
X_test$Stay_In_Current_City_Years <- as.integer(X_test$Stay_In_Current_City_Years)

# converting Gender to binary
X_train$Gender <- ifelse(X_train$Gender == "F", 1, 0)
X_test$Gender <- ifelse(X_test$Gender == "F", 1, 0)

# feature representing the count of each user
user_count <- ddply(X_train, .(User_ID), nrow)
names(user_count)[2] <- "User_Count"
X_train <- merge(X_train, user_count, by="User_ID")
X_test <- merge(X_test, user_count, all.x=T, by="User_ID")

# feature representing the count of each product
product_count <- ddply(X_train, .(Product_ID), nrow)
names(product_count)[2] <- "Product_Count"
X_train <- merge(X_train, product_count, by="Product_ID")
X_test <- merge(X_test, product_count, all.x=T, by="Product_ID")
X_test$Product_Count[is.na(X_test$Product_Count)] <- 0

# feature representing the average Purchase of each product
product_mean <- ddply(X_train, .(Product_ID), summarize, Product_Mean=mean(Purchase))
X_train <- merge(X_train, product_mean, by="Product_ID")
X_test <- merge(X_test, product_mean, all.x=T, by="Product_ID")
X_test$Product_Mean[is.na(X_test$Product_Mean)] <- mean(X_train$Purchase)

# feature representing the proportion of times the user purchases the product more than the product's average
X_train$flag_high <- ifelse(X_train$Purchase > X_train$Product_Mean,1,0)
user_high <- ddply(X_train, .(User_ID), summarize, User_High=mean(flag_high))
X_train <- merge(X_train, user_high, by="User_ID")
X_test <- merge(X_test, user_high, by="User_ID")

train$flag_high <- ifelse(X_train$Purchase > X_train$Product_Mean,1,0)
user_high <- ddply(train, .(User_ID), summarize, User_High=mean(flag_high))
train <- merge(train, user_high, by="User_ID")
X_train <- merge(X_train, user_high, by="User_ID")

###########################My story about the data ###############################
#############################Univariable analysis#################################
library(ggplot2)
#If user high is .5 means the user buy exactly half of the times at more than an average price of the given prod
ggplot(X_train,aes(x=User_High))+geom_density()+ylab('Frequancy')     #Almost a normal distribution 
ggplot(X_train,aes(x=Product_Mean))+geom_density()+ylab('Frequancy')  # This is a multi model distribution 
# This a multy model . thats the respn 


#product_count is the number of times the product is appearing 
ggplot(X_train,aes(x=Product_Count))+xlab('Number of times a product sold')+geom_density() # This distribution is right skewed   
#This means products appearing from 50 to 500 times are more. there are some other products appearing 
# moe than 1000 and 1500 times , butthese are very less


ggplot(X_train,aes(x=Purchase))+geom_density()+ylab('Frequancy')      #Multy model distribution 
ggplot(X_train,aes(x=Occupation))+geom_density()+ylab('Frequancy')    #Multy model distribution 
ggplot(X_train,aes(x=Age))+geom_density()+ylab('Frequancy')

#############################Bi-variable analysis#################################
#You can observe a perfect non-leanior relationship here.
#This show a Coconut leaf kind of structure but a lot more speak about it.
# more frequent shoppers have contributed to more frequent product transactions no doubt about it .
# But product frequency(product count) is bounded by a upper limit of user count. For example product freuncy(fre#quency of transactions) of 500 to 550 is contributed by user count  .6 or lesser(not more than that)
#There are some exceptions, but this is what is the reason behind these shapes!
ggplot(X_train[spl,],aes(y=User_Count,x=Product_Count))+geom_dotplot(binwidth = 1)
#These two are said to be best features since they exhibit best non Leanor relationship 

ggplot(X_train[spl,],aes(x=User_High,fill=Gender))+geom_density()+facet_grid(Gender ~.) #Aproximately normal 

#Nothing more to talk about this

#Purchase distributions for different Occupations looks same but few variations .
# There is nothing to explain more about this.
ggplot(X_train[spl,],aes(x=Purchase,fill=Occupation))+geom_density()+facet_grid(Occupation ~.)



#Users those who purchase more than a product average cost are exist in City C not in cities A and B
pl1<-ggplot(X_train[spl,],aes(y=User_High,x=City_Category_A))+geom_boxplot()+facet_grid(City_Category_A ~.)
pl2<-ggplot(X_train[spl,],aes(y=User_High,x=City_Category_B))+geom_boxplot()+facet_grid(City_Category_B ~.)
pl3<-ggplot(X_train[spl,],aes(y=User_High,x=City_Category_C))+geom_boxplot()+facet_grid(City_Category_C ~.)
grid.arrange(pl1,pl2,pl3,ncol=3)
#Observation by above plot , on an average female customers from city C shope products 53.75% of the time more than their avarage cost.
#Rest all the customers of all cities buy products on an average 50% of the times more than their average cost

#If we take product frequency(Product_count) from 200 to 500 , the city C has higher contribution 
pl1<-ggplot(X_train[spl,],aes(x=Product_Count,fill=City_Category_A))+geom_density()+facet_grid(City_Category_A ~.)
pl2<-ggplot(X_train[spl,],aes(x=Product_Count,fill=City_Category_B))+geom_density()+facet_grid(City_Category_B ~.)
pl3<-ggplot(X_train[spl,],aes(x=Product_Count,fill=City_Category_C))+geom_density()+facet_grid(City_Category_C ~.)
grid.arrange(pl1,pl2,pl3,ncol=3)

#When Zoom more on to the right, the contribution goes to City A for more frequent shoppers for product frequency(Product_count)1500 to 2000
pl1<-ggplot(X_train[spl,],aes(x=Product_Count,fill=City_Category_A))+geom_density()+facet_grid(City_Category_A ~.)+
  scale_x_continuous(limits = c(1500,2000 ))
pl2<-ggplot(X_train[spl,],aes(x=Product_Count,fill=City_Category_B))+geom_density()+facet_grid(City_Category_B ~.)+
  scale_x_continuous(limits = c(1500,2000))
pl3<-ggplot(X_train[spl,],aes(x=Product_Count,fill=City_Category_C))+geom_density()+facet_grid(City_Category_C ~.)+
scale_x_continuous(limits = c(1500,2000))
grid.arrange(pl1,pl2,pl3,ncol=1)

#By observing above two blocks, product that sold 500 time are more observed in City C and products that sold 1500 to 2000 are more observed in A


library(caTools)
set.seed(3000)
spl = sample.split(train$User_ID, SplitRatio = 0.05)
Pdata=train[spl,]
by(Pdata$Purchase,Pdata$Gender,summary)   # Statistical exploration 
#There is no big change inthe median but the mean if males is more than feamles , This means men have some extreme purchases

qplot(x=tenuer,data=Pdata,binwidth=30,color=I('black',fill=I('#099009))  # a monthly view
qplot(x=tenuer/365,data=Pdata,binwidth=1,color=I('black',fill=I('#099009))  # a yearly view
                                                          
qplot(x=tenuer/365,data=Pdata,binwidth=.25,color=I('black'),fill=I('#F79420'))+
scale_x_continues(breaks=seq(1,7,1),limit=c(0,7))  # a yearly view with quarterly bins


install.packages('gridExtra')
library(gridExtra) 
library(ggplot)
#Transformation 
logScale<-qplot(x=log10(Purchase),data=Pdata) # x ais is not scaled by log10
countScale<-ggplot(aes(x=Purchase),data=Pdata)+geom_histogram()+scale_x_log10() # Xaxis is scaled by 10

grid.arrange(logScale,countScale,ncol=2)

#Gender inbalance in the data set
table(Pdata$Gender)    # females are very less compared to males. So lets see the Gender proportions in the shopping scale

# The below frequency poligon says that the proportion of male customer is higher than female in all range of purchases
ggplot(aes(x = Purchase, y = ..count../sum(..count..)), data = subset(Pdata, !is.na(Gender))) + 
  geom_freqpoly(aes(color = Gender), binwidth=10) + 
  scale_x_continuous(limits = c(0, 27723), breaks = seq(0, 27723, 50)) + 
  xlab('Purchases') + 
  ylab('Proportion of customers with that Purchase amount')


#Lets zoom more on the right side , I think some places feamale shoppers overtake males in the case of higher purchases
ggplot(aes(x = Purchase, y = ..count../sum(..count..)), data = subset(Pdata, !is.na(Gender))) + 
  geom_freqpoly(aes(color = Gender), binwidth=10) + 
  scale_x_continuous(limits = c(20000, 27723), breaks = seq(0, 1000, 50)) + 
  xlab('Purchases') + 
  ylab('Proportion of customers with that Purchase amount')


#This is where the feamle proportions overtake teh Male proportions 
ggplot(aes(x = Purchase, y = ..count../sum(..count..)), data = subset(Pdata, !is.na(Gender))) + 
  geom_freqpoly(aes(color = Gender), binwidth=10) + 
  scale_x_continuous(limits = c(1500, 2000), breaks = seq(0, 1000, 50)) + 
  xlab('Purchases') + 
  ylab('Proportion of customers with that Purchase amount')


ggplot(aes(x = User_High, y = ..count../sum(..count..)), data = subset(train[spl,], !is.na(Gender))) + 
  geom_freqpoly(aes(color = Gender), binwidth=50) + 
  scale_x_continuous(limits = c(0,27723), breaks = seq(0,27723,10)) + 
  xlab('user high') + 
  ylab('Proportion of customers with that user high')

#Zoom on the left side
ggplot(aes(x = User_High, y = ..count../sum(..count..)), data = subset(train[spl,], !is.na(Gender))) + 
  geom_freqpoly(aes(color = Gender), binwidth=1) + 
  scale_x_continuous(limits = c(0,10), breaks = seq(0,10,1)) + 
  xlab('user high') + 
  ylab('Proportion of customers with that user high')



#Box PLots
#Even though there are more outliers in the feamle box plot, their proportion is less compared to male in the case of higher purchases. 
qplot(x=Gender, y=Purchase,data=subset(Pdata, !is.na(Gender)),geom='boxplot')

#Outlier : is 1.5 times IQR distance from th emedian
qplot(x=Gender, y=Purchase,data=subset(Pdata, !is.na(Gender)),geom='boxplot',ylim=c(0,15750))  # set the limits to remove the outliers
#so when we see thse plots it is clear that teh female customers shope less than the male

qplot(x=Gender, y=Purchase,data=subset(Pdata, !is.na(Gender)),geom='boxplot')+
scale_y_continuous(limits=c(0,15750))  # set the limits to remove the outliers
#To match the graph details with the above statistics use the coord_cartesian function

by(Pdata$Purchase,Pdata$Gender,summary)  # to summarize the Purchases by Gender
qplot(x=Gender, y=Purchase,data=subset(Pdata, !is.na(Gender)),geom='boxplot')+coord_cartesian(ylim=c(0,15750))
#Coord cartesian will not leave nay observations to summarize , they will be just left while displaying on the graph

