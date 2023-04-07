# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)

# Data Import and Cleaning
gss_data <- read_spss("../data/GSS2016.sav") #read_sav function not found without library call, despite haven listed as part of the tidyverse at https://haven.tidyverse.org/index.html. Reads into a tibble,
gss_tbl <- #tibble(gss_data) %>% Second tibble not necessary given the read_spss function from haven. 
  ##remove all SPSS attributes
  zap_missing(gss_data) %>% #removes all unique characters for SPSS missing data. In haven package
  #labelled::remove_labels() %>% #removes labels of SPSS data
  #labelled::remove_attributes("display_width") %>% #removes display settings for SPSS data. could also just shove the predictor into mutate(workhours = as.numeric(workhours))
  rename(workhours = HRS1) %>% #could also use variable mosthrs
  drop_na(workhours) %>% #1646 cases matches documentation on page 123.
  select(workhours, which(colMeans(is.na(gss_data)) <= 0.25)) %>%
  mutate(workhours = as.numeric(workhours))#selecting all columns WHICH have 25% or less missing data. colMeans() used to generate a proportion of NA per column. 
#colMeans(is.na(gss_tbl)) #quick check that displays all included variables with % missingness, all less than .25

#Visualization
histogram(gss_tbl$workhours, main = "Distribution of workhours")

# Machine Learning Models
set.seed(25)
##split dataset: 75% Train 25% for Test, even distribution of prediction criterion
index <- createDataPartition(gss_tbl$workhours, p = 0.75, list = FALSE) #could also use index <- sample(1:nrow(gss_tbl), .75*nrow(gss_tbl))
train_tbl <- gss_tbl[index, ]
holdout_tbl <- gss_tbl[-index, ]
fold_indices = createFolds(train_tbl$workhours, k = 10) #creates the same 10 folds for every model to train with, allowing fair model comparison
#expanded_grid <- expand.grid(alpha = c(0,0.5, 1), lambda = seq(0.0001, 0.1, length = 10)) #alpha set to try a pure ridge regression model, a balanced mixed model, and a pure lasso regression model

## OLS Regression Model
linear_model <- train(
  workhours ~ .,  #model you want to predict
  data = train_tbl, #Training Dataset
  method = "lm", #OLS Regression
  na.action = "na.pass", #ensures function does not fail when missing data is encountered. 
  preProcess = "medianImpute", #medianImpute required by project
  trControl = trainControl(
    method = "cv", #cross-validation training method
    indexOut = fold_indices, #set to 10 folds above, same folds across models
    verboseIter = TRUE
  )
)
linear_model
linear_predict <- predict(linear_model, holdout_tbl, na.action=na.pass)
r2_linear_holdout <- cor(holdout_tbl$workhours, linear_predict)^2
r2_linear_holdout

## Elastic Net Model
en_model <- train(
  workhours ~ .,  #see above for notes of repeated columns
  data = train_tbl, 
  method = "glmnet", # runs Elastic Net model
  tuneLength = 3, #expands both hyperparameters for this model, as designated above. Can also use tuneGrid = expanded_grid from above
  na.action = "na.pass", 
  preProcess = "medianImpute",
  trControl =  trainControl(
    method = "cv",
    indexOut = fold_indices,
    verboseIter = TRUE
  )
)
en_model
en_predict <- predict(en_model, holdout_tbl, na.action=na.pass)
r2_en_holdout <- cor(holdout_tbl$workhours, en_predict)^2
r2_en_holdout

## Random Forest Model
rf_model <- train(
  workhours ~ ., #see above for notes of repeated columns
  data = train_tbl, 
  tuneLength = 5, #minimnum number of values per hyperparameter run. 3 hyperparameters total
  na.action = "na.pass", 
  preProcess = "medianImpute", 
  method = "ranger", #runs random forest model
  trControl = trainControl(
    method = "cv", 
    indexOut = fold_indices,
    verboseIter = TRUE
  )
)

rf_model
rf_predict <- predict(rf_model, holdout_tbl, na.action=na.pass)
r2_rf_holdout <- cor(holdout_tbl$workhours, rf_predict)^2
r2_rf_holdout

## eXtreme Gradient Boosting Model
gb_model <- train(
  workhours ~ .,  #see above for notes of repeated columns
  data = train_tbl, #see above for notes of repeated columns
  method = "xgbLinear", #1 of 3 extreme Gradient Boosting models. Less hyperparamters than xgbDART or xgbTree, and therefore is expected to run faster, given the tuneLength argument below. xgbDART and xgbTree (similar to Random Forest models) issue warnings (not errors) of using a deprecated hyperparameter `ntree_limit`. 
  tuneLength = 3, #3 selected as a balance of granular hyperparameter values and speed. 2 would be faster, 5 would be slower. 10 took a very, very long time (due to number of parameters).
  na.action = "na.pass",
  preProcess = "medianImpute",
  trControl =  trainControl(
    method = "cv", #could also set to "adaptive_cv" to increase speed and efficiency for all models, but not selected as the project did not call for this specifically
    indexOut = fold_indices,
    verboseIter = TRUE
  )
)
gb_model
gb_predict <- predict(gb_model, holdout_tbl, na.action=na.pass)
r2_gb_holdout <- cor(holdout_tbl$workhours, gb_predict)^2
r2_gb_holdout

# Publication
model_list <- list(Linear = linear_model, ElasticNet = en_model, RandomForest = rf_model, GradientBoosting = gb_model)
results <- summary(resamples(model_list), metric="Rsquared")
dotplot(resamples(model_list), metric="Rsquared", main = "10-Fold CV Rsquared")
results

## Create tibble
cv_rsq <- results$statistics$Rsquared[,"Mean"] #mean values used because they correspond with each selected model, which minimizes RMSEA, as described by the each model output
ho_rsq <- c(r2_linear_holdout, r2_en_holdout, r2_rf_holdout, r2_gb_holdout)
table1_tbl <- tibble(algo = results$models, cv_rsq, ho_rsq) %>%
  mutate(cv_rsq = str_remove(format(round(cv_rsq, 2), nsmall = 2), "^0"),
         ho_rsq = str_remove(format(round(ho_rsq, 2), nsmall = 2), "^0"))
table1_tbl

## 1. How did your results change between models? Why do you think this happened, specifically?
## Results were different between these models as a result of the underlying assumptions made by each model, including the subsequent values for the hyperparameters used in each model. For example, some of these models, like the linear model and elastic net, assume workhours shares a linear relationship with the predictor variables, and apply varying degrees of penalties for model complexity. On the other hand, the random forest model makes no such assumptions about linearity. 

## 2. How did you results change between k-fold CV and holdout CV? Why do you think this happened, specifically?
# Model fit reduced pretty dramatically from the k-fold CV to the holdout CV. This suggests that all of our models are overfitting to some extent, using the training subset of our data, despite our 10-fold model parameter estimation process. 

## 3. Among the four models, which would you choose for a real-life prediction problem, and why? Are there tradeoffs? Write up to a paragraph.

## In a real-life prediction problem, model selection would depend on the situation! Before picking a model, we should consider the research question to answer, the underlying data quantity and complexity, any assumptions made about the relationships of interest, how urgently we need an answer, permitted computational expense, and the extent to which we would need to be transparent or explain the model to others (as a "black box" solution is not always appropriate). These are some of the tradeoffs we have to make when selecting and deploying a particular model. If I had to make a choice from these 4 models without additional context, I would select the Random Forest model, as it is a flexible model that balances run time, computational expense, and prediction accuracy for a wide variety of research questions.