# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)

# Data Import and Cleaning
gss_data <- read_spss("../data/GSS2016.sav") #read_sav function not found without library call, despite haven listed as part of the tidyverse at https://haven.tidyverse.org/index.html
gss_tbl <- tibble(gss_data) %>%
  ##remove all SPSS attributes
  zap_missing() %>% #removes all unique characters for SPSS missing data. In haven package
  labelled::remove_labels() %>% #removes labels of SPSS data
  labelled::remove_attributes("display_width") %>% #removes display settings for SPSS data
  rename(workhours = HRS1) %>%
  drop_na(workhours) #1646 cases matches documentation on page 123.
gss_tbl <- select(gss_tbl, which(colMeans(is.na(gss_tbl)) <= 0.25)) #selecting all columns WHICH have 25% or less missing data. colMeans() used to generate a proportion of NA per column. 
#colMeans(is.na(gss_tbl)) #quick check that displays all included variables with % missingness, all less than .25

#Visualization
histogram(gss_tbl$workhours)

# Machine Learning Models
set.seed(25)
##split dataset: 75% Train 25% for Test, even distribution of 
index <- createDataPartition(gss_tbl$workhours, p = 0.75, list = FALSE)
train_data <- gss_tbl[index, ]
test_data <- gss_tbl[-index, ]

## OLS Regression Model
LM_model <- train(
  workhours ~ .,  #model you want to predict
  data = train_data, #Training Dataset
  method = "lm", #OLS Regression
  na.action = "na.pass", #passes over columns where medians cannot be computed
  preProcess = "medianImpute",
  trControl = trainControl(
    method = "cv", 
    number = 10, # number of folds
    verboseIter = TRUE
  )
)
LM_model

## Elastic Net Model
ElasticNet_model <- train(
  workhours ~ .,  #model you want to predict
  data = train_data, #Training Dataset
  method = "glmnet",
  tuneLength = 10,
  na.action = "na.pass", 
  preProcess = "medianImpute",
  trControl =  trainControl(
    method = "cv", 
    number = 10,
    verboseIter = TRUE
  )
)
ElasticNet_model

## Random Forest Model
RandomForest_model <- train(
  workhours ~ ., #model you want to predict
  data = train_data, #Training dataset
  tuneLength = 5, #minimnum number of nodes
  na.action = "na.pass", 
  preProcess = "medianImpute",
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 10, 
    verboseIter = TRUE
  )
)
RandomForest_model
## eXtreme Gradient Boosting Model
GradientBoosting_model <- train(
  workhours ~ .,  #model you want to predict
  data = train_data, #Training Dataset
  method = "xgbLinear",
  tuneLength = 3,
  na.action = "na.pass", 
  preProcess = "medianImpute",
  trControl =  trainControl(
    method = "cv", 
    number = 10,
    verboseIter = TRUE
  )
)
GradientBoosting_model

# Publication

# Create model_list
model_list <- list(item1 = LM_model)
# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)
# Summarize the results
summary(resamples)
