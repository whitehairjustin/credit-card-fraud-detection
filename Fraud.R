#libraries needed
require(dplyr)
require(randomForest)
require(ROCR)

#read data
data = read.csv("Fraud/Fraud_Data.csv")
ip_addresses = read.csv("Fraud/IpAddress_to_Country.csv")

str(data)
str(ip_addresses)

#are there duplicates?
nrow(data) == length(unique(data$user_id))

#Let's add the country to the original data set by using the ip address
data_country = rep(NA, nrow(data))

for (i in 1: nrow(data))
  {
  tmp = as.character(ip_addresses [data$ip_address[i] >= ip_addresses$lower_bound_ip_address & data$ip_address[i] <= ip_addresses$upper_bound_ip_address,
                                   "country"])
  if (length(tmp) == 1) {data_country[i] = tmp}
}

data$country = data_country

data[, "signup_time"] = as.POSIXct(data[, "signup_time"], tz="GMT")
data[, "purchase_time"] = as.POSIXct(data[, "purchase_time"], tz="GMT")

summary(data)
summary(as.factor(data$country))

##################################### Fearure Engineering ##################################

# Time difference between sign-up time and purchase time
# If the device id is unique or certain users are sharing the same device (many different user ids using
# the same device could be an indicator of fake accounts)
# Same for the ip address. Many different users having the same ip address could be an indicator of
# fake accounts.
# Usual week of the year and day of the week from time variables

#time difference between purchase and signup
data$purchase_signup_diff = as.numeric(difftime(as.POSIXct(data$purchase_time, tz="GMT"), as.POSIXct(data$signup_time, tz="GMT"), unit="secs"))

#check for each device id how many different users had it
data = data %>%
  group_by(device_id) %>%
  mutate (device_id_count = n())

#check for each ip address how many different users had it
data = data.frame(data %>%
                    group_by(ip_address) %>%
                    mutate (ip_address_count = n()))

#day of the week
data$signup_time_wd = format(data$signup_time, "%A")
data$purchase_time_wd = format(data$purchase_time, "%A" )

#week of the yr
data$signup_time_wy = as.numeric(format(data$signup_time, "%U"))
data$purchase_time_wy = as.numeric(format(data$purchase_time, "%U" ))

#data set for the model. Drop first 3 vars and device id.
data_rf = data[, -c(1:3, 5)]

#replace the NA in the country var
data_rf$country[is.na(data_rf$country)]="Not_found"

#just keep the top 50 country, everything else is "other"
data_rf$country = ifelse(data_rf$country %in% names(sort(table(data_rf$country), decreasing = TRUE ) )[51:length(unique(data_rf$country))], # after top 50 countries
                         "Other", as.character(data_rf$country)
)

#make class a factor
data_rf$class = as.factor(data_rf$class)

#all characters become factors
data_rf[sapply(data_rf, is.character)] <- lapply(data_rf[sapply(data_rf, is.character)], as.factor)

# Class of target variable
y <- table(data_rf$class)
round(prop.table(y) * 100, digits = 1)
## 9.4% Fraud Rate

# write.csv(data_rf, file = "fraud.csv")
# data_rf <- read.csv("fraud.csv")

# train/test split

data_rf$class <- as.factor(data_rf$class)
train_sample = sample(nrow(data_rf), size = nrow(data)*0.80)
train_data = data_rf[train_sample,]
Validation_data = data_rf[-train_sample,]
test_data <- Validation_data

####################### Model building #########################3

#rf = randomForest(y=train_data$class, x = train_data[, -7],
#                  ytest = test_data$class, xtest = test_data[, -7],
#                  ntree = 50, mtry = 3, keep.forest = TRUE)
#rf

library(mlbench)
library(caret)
library(pROC)

################# Random Forests
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(train_data))
tunegrid <- expand.grid(.mtry=mtry)
rfDefault <- train(class~., data=train_data, method="rf", metric=metric, tuneGrid=tunegrid,
                   trControl=trainControl)
print(rfDefault)

pred_rf <- predict(rfDefault,newdata = test_data, type = "prob")
plot(roc(test_data$class,pred_rf[,1]))
auc(roc(test_data$class,pred_rf[,1]))

### Tuning mtry parameter
# Random search
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
mtry <- sqrt(ncol())
rfRandom <- train(Class~., data=dataset, method="rf", metric=metric, tuneLength=15,
                  trControl=trainControl)
print(rfRandom)
plot(rfRandom)

# Grid search
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15))
rfGrid <- train(Class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid,
                trControl=trainControl)

colnames(train_data)

# Variable importance
varImpPlot(rfRandom,type=2)

# Partial dependence plots with 4 most important variables
op <- par(mfrow=c(2, 2))
partialPlot(rfRandom, train_data, purchase_signup_diff, 1)
partialPlot(rfRandom, train_data, purchase_time_wy, 1)
partialPlot(rfRandom, train_data, ip_address_count, 1)
partialPlot(rfRandom, train_data, device_id_count, 1)


######### Boosting algorithms #########

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
# C5.0
set.seed(seed)
fit.c50 <- train(Class~., data=train_data, method="C5.0", metric=metric,
                 trControl=trainControl)
# Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(Class~., data=train_data, method="gbm", metric=metric,
                 trControl=trainControl, verbose=FALSE)

# summarize results
boostingResults <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boostingResults)
dotplot(boostingResults)

pred_gbm <- predict(fit.gbm,newdata = test_data, type = "prob")
plot(roc(test_data$class,pred_gbm[,1]))
auc(roc(test_data$class,pred_gbm[,1]))
