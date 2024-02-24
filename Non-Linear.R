# Saeed Peyman/ Usama Ayub
# MA 5790 - Predictive Modeling
# Presentation II

library(AppliedPredictiveModeling)
library(mda)
library(MASS)
library(caret)
library(klaR)
library(e1071)
library(caret)
library(earth)
library(Formula)
library(plotmo)
library(plotrix)
library(TeachingDemos)

# detach("mda", unload = TRUE)
# detach("earth", unload = TRUE)


setwd('/Users/usamaayub/Downloads/MTU MS Data Science/2nd Semester(Fall23)/MA5790/Project')
getwd()

data <- read.csv('clean_data.csv')
data$diagnosis <- factor(data$diagnosis)

# *** Info about our data ***
names(data)
summary(data)
head(data)
str(data)
dim(data)
attach(data)

names(data)



# Data Splitting:
set.seed(1234)
trainIndex <- createDataPartition(data$diagnosis, p = .8, list = FALSE)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

# 10 Fold Cross Validation
ctrl <- trainControl(method = "cv", number = 10)
levels(trainData$diagnosis) <- make.names(levels(trainData$diagnosis))
levels(testData$diagnosis) <- make.names(levels(testData$diagnosis))

# *** Linear Model Building ***

############ Logistic Regression ###############

## The glm function (for GLMs) in base R is commonly used to fit 
## logistic regression models. The syntax is


set.seed(975)

param_grid <- expand.grid(lambda = seq(0, 1, by = 0.01))


# Rename the lambda column to parameter
colnames(param_grid) <- c("parameter")

model_lr <- train(x=predictors,
                  y = response,
                  method = "glm",
                  metric = "roc",
                  trControl = ctrl,
                  tuneGrid = param_grid)
model_lr
#plot(model_lr)

####################################################

library(caret)
library(glmnet)

# Define the grid of tuning parameters
param_grid <- expand.grid(alpha = 1, # for lasso; use 0 for ridge
                          lambda = seq(0, 1, by = 0.1))

# Train the model
model_lr1 <- train(x = predictors,
                   y = response,
                   method = "glmnet",
                   metric = "roc",
                   trControl = ctrl,
                   tuneGrid = param_grid)


model_lr1





####################### glm model #################

# Apply Logistic Regression Model usin glm for better visualisation

model_lr2 <- glm(data$diagnosis ~ ., data = data, family = "binomial")
model_lr2

summary(model_lr2)


prediction_lr <- predict(model_lr2, newdata = test_data, type = "response")
confusion_matrix_lr <- table(Actual = test_data$diagnosis, Predicted = ifelse(prediction_lr > 0.5, 1, 0))
accuracy_lr <- sum(diag(confusion_matrix_lr)) / sum(confusion_matrix_lr)
print(confusion_matrix_lr)
print(paste("Accuracy of LR model:", accuracy_lr, "%"))


# Make Predictions on the Test Set
predictions <- predict(model_lr2, newdata = test_data, type = "response")

confusion_matrix <- table(Actual = test_data$diagnosis, Predicted = ifelse(predictions > 0.5, 1, 0))
#confusion_matrix <- table(Actual = test_data$diagnosis, Predicted = predictions)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
print(paste("Accuracy of LR model:", accuracy, "%"))



## Function to create confusion matrix plot

plot_confusion_matrix <- function(conf_matrix, title) {
  ggplot() +
    geom_tile(aes(x = Var2, y = Var1, fill = Freq), data = as.data.frame(as.table(conf_matrix)), color = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    theme_minimal() +
    labs(title = title, x = "Predicted", y = "Actual") +
    theme(axis.text = element_text(size = 10), axis.title = element_text(size = 12))
}
## Confusion matrix plot for Logistic Regression

plot_roc_curve <- function(predictions, true_labels, title) {
  pred <- prediction(predictions, true_labels)
  perf <- performance(pred, "tpr", "fpr")
  
  ggplot(data = data.frame(x = perf@x.values[[1]], y = perf@y.values[[1]])) +
    geom_line(aes(x = x, y = y), color = "steelblue", size = 1.2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    labs(title = title, x = "False Positive Rate (FPR)", y = "True Positive Rate (TPR)") +
    theme_minimal() +
    theme(axis.text = element_text(size = 10), axis.title = element_text(size = 12))
}


############ Linear Discriminant Analysis ###############

#response1 <- as.factor(response)

# Check the levels of the factor
levels(response1)


set.seed(900)
LDA_Fit <- train(
  x = predictors,      # Predictor variables
  y = response1,      # Response variable
  method = "lda",     # LDA method
  metric = "roc",   # Evaluation metric (you can change it based on your preference)
  trControl = ctrl    # Training control settings (if defined elsewhere in your code)
)

LDA_Fit


# Use the same 'train_data' and 'test_data' obtained from the previous steps

# Extract response variable and predictor variables


length(response1)
dim(predictors)
# Step 2: Linear Discriminant Analysis (LDA) Model
lda_model <- lda(response1 ~ ., data = predictors)
lda_model
# Print summary of the LDA model
summary(lda_model)

# Step 3: Make Predictions on the Test Data
lda_predictions <- predict(lda_model, newdata = test_data[, -1])

# Step 4: Evaluate LDA Model
conf_matrix_lda <- table(lda_predictions$class, test_data$diagnosis)
accuracy_lda <- sum(diag(conf_matrix_lda)) / sum(conf_matrix_lda)

# Print confusion matrix and accuracy
print("Confusion Matrix for LDA Model:")
print(conf_matrix_lda)
print(paste("Accuracy of LDA Model: ", round(accuracy_lda * 100, 2), "%"))

plot_confusion_matrix(conf_matrix_lda, "Confusion Matrix - LDA")





################ PLSDA###############

set.seed(900)
PLSDA_model <- train(x = predictors,
                     y = response,
                     method = "pls",
                     tuneGrid = expand.grid(.ncomp = 1:4),
                     # preProc = c("center","scale"),
                     metric = "Rsquared",
                     trControl = ctrl)
PLSDA_model
plot(PLSDA_model)

test_data$diagnosis <- as.factor(test_data$diagnosis)

pred_PLSDA <-predict(PLSDA_model,test_data)
confusion_matrix_PLSDA <- table(Actual = test_data$diagnosis, Predicted = ifelse(pred_PLSDA > 0.5, 1, 0))



# Accuracy, Kappa will printed in Confusion Matrix and Statistics
print(confusion_matrix_PLSDA)


########### Penalized Models ###########

# *** Penalized Model ***
# The glmnet package in R is used to fit penalized models. 
# Here we will use logistic regression with both L1 (Lasso) and L2 (Ridge) penalties.

set.seed(123)

# Define the grid of tuning parameters
param_grid <- expand.grid(alpha = 1, # Alpha = 1 corresponds to the Lasso penalty
                          lambda = seq(0, 1, by = 0.1)) # Lambda is the tuning parameter

# Fit the penalized logistic regression model
model_penalized <- train(x = predictors,
                         y = response,
                         method = "glmnet",
                         metric = "Rsquared",
                         trControl = ctrl,
                         tuneGrid = param_grid)

# Print the model
print(model_penalized)

predictions_pm <- predict(model_penalized, newdata = test_data)

# Create Confusion Matrix
confusion_matrix_pm <- table(Actual = test_data$diagnosis, Predicted = ifelse(predictions_pm > 0.5, 1, 0))

# Print Confusion Matrix
print(confusion_matrix_pm)

# Calculate Accuracy
accuracy <- sum(diag(confusion_matrix_pm)) / sum(confusion_matrix_pm)
print(paste("Accuracy of Penalized Model:", accuracy, "%"))

# Plot the Model
plot(model_penalized)



########### ***Nearest Shrunken Centroids*** ###########


nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))
set.seed(476)
nscFit <- train(x = predictors,
                y = response,
                method = "glmnet",
                # preProc = c("center", "scale"),
                #tuneGrid = nscGrid,
                metric = "Rsquared",
                trControl = ctrl)
nscFit
plot(nscFit)
predictionNSC <-predict(nscFit,test_data)

summary(predictionNSC)


str(train_data)

############################################################






# ------------------------Non-Linear Models--------------------------
# ***Non-Linear Classification Models**:

# - **Non-Linear Discriminant Analysis**: Extension of LDA for non-linear relationships.
# - **Neural Networks**: As in regression, suitable for complex classification tasks.
# - **Flexible Discriminant Analysis**: Combines discriminant analysis with non-linear transformations.
# - **Support Vector Machines (SVM)**: Effective in high dimensional spaces.
# - **K-Nearest Neighbors (KNN)**: Classifies based on the majority vote of its neighbors.
# - **Naive Bayes**: Based on applying Bayes' theorem with strong independence assumptions.

# Pre-processing
preProcValues <- preProcess(trainData, method = c("center", "scale"))
trainData <- predict(preProcValues, trainData)


# 1. *** Flexible Discriminant Analysis ***
set.seed(1234)
grid_fda <- expand.grid(.degree = 1:2, .nprune = 2:38)
fdaFit <- caret::train(diagnosis ~ . ,
                         data = trainData,
                         method = "fda",
                         metric = "roc",
                         tuneGrid = grid_fda,
                         trControl = trainControl(method = "cv"))

fdaFit

fdaFit_pred <- predict(fdaFit,trainData)
levels_trainData <- levels(trainData$diagnosis)
levels_fdaFit_pred <- levels(fdaFit_pred)


# Make sure levels are consistent
if (!identical(levels_trainData, levels_fdaFit_pred)) {
  # Update the levels of madFit_pred to match those of trainData
  fdaFit_pred <- factor(fdaFit_pred, levels = levels_trainData)
}
confusionMatrix(fdaFit_pred, trainData$diagnosis)

plot(fdaFit)


# 2. *** Support Vector Machines ***
svmGrid <- expand.grid(C = c(0.1, 1, 10, 100, 1000), 
                       sigma = c(1, 0.1, 0.01, 0.001, 0.0001))
set.seed(1234)
svmFit <- train(diagnosis ~ ., 
                data = trainData, 
                method = "svmRadial",
                metric = "roc",
                tuneGrid = svmGrid, 
                preProc = c("center", "scale", "spatialSign"),
                trControl = ctrl)

svmFit

svmFit_pred <- predict(svmFit,trainData)
levels_trainData <- levels(trainData$diagnosis)
levels_svmFit_pred <- levels(svmFit_pred)


# Make sure levels are consistent
if (!identical(levels_trainData, levels_svmFit_pred)) {
  # Update the levels of madFit_pred to match those of trainData
  svmFit_pred <- factor(svmFit_pred, levels = levels_trainData)
}
confusionMatrix(svmFit_pred, trainData$diagnosis)


plot(svmFit)


# 3. *** Mixture Discriminant Analysis ***
set.seed(1234)
mdaFit <- train(diagnosis ~ .,
                data = trainData,
                method = "mda",
                metric = "roc",
                tuneGrid = expand.grid(.subclasses = 1:10),
                trControl = ctrl)


mdaFit

mdaFit_pred <- predict(mdaFit,trainData)
levels_trainData <- levels(trainData$diagnosis)
levels_mdaFit_pred <- levels(mdaFit_pred)


# Make sure levels are consistent
if (!identical(levels_trainData, levels_mdaFit_pred)) {
  # Update the levels of madFit_pred to match those of trainData
  mdaFit_pred <- factor(mdaFit_pred, levels = levels_trainData)
}
confusionMatrix(mdaFit_pred, trainData$diagnosis)


plot(mdaFit)

# 4. *** Neural Networks ***
set.seed(1234)
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (96 + 1) + (maxSize+1)*3)
nnetFit <- train(diagnosis ~ . ,
                 data = trainData,
                 method = "nnet",
                 metric = "roc",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit

nnetFit_pred <- predict(nnetFit,trainData)
levels_trainData <- levels(trainData$diagnosis)
levels_nnetFit_pred <- levels(nnetFit_pred)


# Make sure levels are consistent
if (!identical(levels_trainData, levels_nnetFit_pred)) {
  # Update the levels of madFit_pred to match those of trainData
  nnetFit_pred <- factor(nnetFit_pred, levels = levels_trainData)
}
confusionMatrix(nnetFit_pred, trainData$diagnosis)

plot(nnetFit)


# 5. *** K-Nearest Neighbors ***
set.seed(1234)
knnFit <- train(diagnosis ~ ., 
                  data = trainData,
                  method = "knn",
                  metric = "roc",
                  preProc = c("center", "scale"),
                  tuneGrid = data.frame(.k = 1:50),
                  trControl = ctrl)

knnFit

knnFit_pred <- predict(knnFit,trainData)
levels_trainData <- levels(trainData$diagnosis)
levels_knnFit_pred <- levels(knnFit_pred)


# Make sure levels are consistent
if (!identical(levels_trainData, levels_knnFit_pred)) {
  # Update the levels of madFit_pred to match those of trainData
  knnFit_pred <- factor(knnFit_pred, levels = levels_trainData)
}
confusionMatrix(knnFit_pred, trainData$diagnosis)
plot(knnFit)

# 6. *** Naive Bayes ***
set.seed(1234)
tuneGridNB <- expand.grid(.fL = seq(0, 3, by = 1),
                          .usekernel = TRUE,
                          .adjust = seq(0.5, 1.5, by = 0.5))  

nbFit <- train(diagnosis ~ ., 
                  data = trainData,
                  method = "nb",
                  metric = "roc",
                  tuneGrid = tuneGridNB,
                  trControl = ctrl)

nbFit

nbFit_pred <- predict(nbFit,trainData)
levels_trainData <- levels(trainData$diagnosis)
levels_nbFit_pred <- levels(nbFit_pred)


# Make sure levels are consistent
if (!identical(levels_trainData, levels_nbFit_pred)) {
  # Update the levels of madFit_pred to match those of trainData
  nbFit_pred <- factor(nbFit_pred, levels = levels_trainData)
}
confusionMatrix(nbFit_pred, trainData$diagnosis)

plot(nbFit)


# Best Model Awards: 
#### 1. SVM	sigma = 0.0001, C = 1000	0.967
#### 2. NN	size = 1, decay = 2	0.9577
#### 3. MDA	subclass = 2	0.9576
#### 4. FDA	degree = 1, nprune = 16	0.9529
#### 5. KNN	k = 4	0.9206
#### 6. NB	fL = 0, usekernel = TRUE, adjust = 1	0.8723


# Testing on 2 best Models

# *** Support Vector Machines ***

svmFit_pred_test <- predict(svmFit,testData)
levels_testData <- levels(testData$diagnosis)
levels_svmFit_pred_test <- levels(svmFit_pred_test)


# Make sure levels are consistent
if (!identical(levels_testData, levels_svmFit_pred_test)) {
  # Update the levels of madFit_pred to match those of testData
  svmFit_pred_test <- factor(svmFit_pred_test, levels = levels_testData)
}
confusionMatrix(svmFit_pred_test, testData$diagnosis)


# *** Neural Networks ***
nnetFit_pred_test <- predict(nnetFit,testData)
levels_testData <- levels(testData$diagnosis)
levels_nnetFit_pred_test <- levels(nnetFit_pred_test)


# Make sure levels are consistent
if (!identical(levels_testData, levels_nnetFit_pred_test)) {
  # Update the levels of madFit_pred to match those of testData
  nnetFit_pred_test <- factor(nnetFit_pred_test, levels = levels_testData)
}
confusionMatrix(nnetFit_pred_test, testData$diagnosis)

# SVM model with sigma = 0.0001 and C = 1000 stands out as the best model
# for this dataset, achieving the highest Kappa score (0.9227) and demonstrating strong
# predictive performance.

# Important Predictors in SVM
importance <- varImp(svmFit, scale = FALSE)
importance

# Plot of top 5 most important predictors in SVM
plot(importance,top=5)
