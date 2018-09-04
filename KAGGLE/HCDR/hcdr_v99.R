library(data.table)
install.packages('sqldf')
library('sqldf')
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



missing_data=as.data.frame(is.na(my_train[,1:122]))
for (i in 1:122){
  missing_data[,i]=as.numeric(missing_data[,i])
}

missing_data$TARGET<-my_train$TARGET

missing_data$TARGET=as.factor(missing_data$TARGET)
#library(caret)
#model_gbm<-train(missing_data[,c(-2)],missing_data[,c(2)],method='gbm')

library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(missing_data)
#test.h2o <- as.h2o(c.test)
y.dep <- 2
x.indep <- c(1,3:122)


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




#-----------------------------

missing_test=as.data.frame(is.na(test[,1:121]))
for (i in 1:121){
  missing_test[,i]=as.numeric(missing_test[,i])
}

missing_test.h2o=as.h2o(missing_test)

finalgbm_predictions<-h2o.predict(
  object = gbm
  ,newdata = missing_test.h2o)

Predictions=as.data.frame(finalgbm_predictions)
test$TARGET=Predictions$p0
write.table(test[,c(1,122)],'mispredminus.csv',row.names=FALSE,col.names=TRUE,sep=",")
#-----------------------------------------------




hyper_params = list( 
  ## restrict the search to the range of max_depth established above
  max_depth = seq(minDepth,maxDepth,1),                                      
  
  ## search a large space of row sampling rates per tree
  sample_rate = seq(0.2,1,0.01),                                             
  
  ## search a large space of column sampling rates per split
  col_sample_rate = seq(0.2,1,0.01),                                         
  
  ## search a large space of column sampling rates per tree
  col_sample_rate_per_tree = seq(0.2,1,0.01),                                
  
  ## search a large space of how column sampling per split should change as a function of the depth of the split
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),                      
  
  ## search a large space of the number of min rows in a terminal node
  min_rows = 2^seq(0,log2(nrow(train))-1,1),                                 
  
  ## search a large space of the number of bins for split-finding for continuous and integer columns
  nbins = 2^seq(4,10,1),                                                     
  
  ## search a large space of the number of bins for split-finding for categorical columns
  nbins_cats = 2^seq(4,12,1),                                                
  
  ## search a few minimum required relative error improvement thresholds for a split to happen
  min_split_improvement = c(0,1e-8,1e-6,1e-4),                               
  
  ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")       
)

search_criteria = list(
  ## Random grid search
  strategy = "RandomDiscrete",      
  
  ## limit the runtime to 60 minutes
  max_runtime_secs = 3600,         
  
  ## build no more than 100 models
  max_models = 100,                  
  
  ## random number generator seed to make sampling of parameter combinations reproducible
  seed = 1234,                        
  
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 5,                
  stopping_metric = "AUC",
  stopping_tolerance = 1e-3
)



grid <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  
  ## hyper-parameter search configuration (see above)
  search_criteria = search_criteria,
  
  ## which algorithm to run
  algorithm = "gbm",
  
  ## identifier for the grid, to later retrieve it
  grid_id = "final_grid", 
  
  ## standard model parameters
  x = predictors, 
  y = response, 
  training_frame = train, 
  validation_frame = valid,
  
  ## more trees is better if the learning rate is small enough
  ## use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.05,                                                         
  
  ## learning rate annealing: learning_rate shrinks by 1% after every tree 
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,                                               
  
  ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
  max_runtime_secs = 3600,                                                 
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,                                                
  
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234                                                             
)
