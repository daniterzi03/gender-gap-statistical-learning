# Gender Gap Classification with Statistical Learning
# Portfolio version of the project workflow
# Author: Daniele Terzi

library(readxl)
library(dplyr)
library(caret)
library(MASS)
library(glmnet)
library(e1071)
library(pROC)
library(gam)

# Data -----------------------------------------------------------------------
df <- read_excel("data/gender_gap.xlsx", sheet = "Data")

# Missing-value treatment used in the project
for (v in c("services", "manufactoring", "industry",
            "GDP_growth", "GDP_per_capita")) {
  df[[v]][is.na(df[[v]])] <- median(df[[v]], na.rm = TRUE)
}

# Outcome --------------------------------------------------------------------
df <- df %>%
  mutate(
    ratio_FM = log(employers_female / employers_male),
    industry_gap = industry_male - industry_female,
    services_gap = services_male - services_female,
    lf_gap = labor_force_male - labor_force_female,
    wage_gap = wage_salaried_male - wage_salaried_female,
    school_gap = school_male - school_female,
    life_gap = life_male - Life_female,
    trade_openness = trade,
    serv_vs_ind = services - industry,
    development_index = GDP_per_capita * Life_female
  )

threshold <- quantile(df$ratio_FM, 0.60, na.rm = TRUE)
df$y <- factor(ifelse(df$ratio_FM < threshold, "Yes", "No"))

model_df <- df %>%
  select(
    y, GDP_growth, industry_gap, services_gap, lf_gap, wage_gap,
    school_gap, life_gap, trade_openness, serv_vs_ind, development_index
  )

# Train/test split -----------------------------------------------------------
set.seed(1234)
idx <- createDataPartition(model_df$y, p = 0.75, list = FALSE)
train <- model_df[idx, ]
test  <- model_df[-idx, ]

# Logistic regression --------------------------------------------------------
logit <- glm(y ~ ., data = train, family = binomial)
p_logit <- predict(logit, newdata = test, type = "response")

# LDA / QDA ------------------------------------------------------------------
lda_fit <- lda(y ~ ., data = train)
qda_fit <- qda(y ~ ., data = train)

p_lda <- predict(lda_fit, test)$posterior[, "Yes"]
p_qda <- predict(qda_fit, test)$posterior[, "Yes"]

# Naive Bayes ----------------------------------------------------------------
nb_fit <- naiveBayes(y ~ ., data = train)
p_nb <- predict(nb_fit, test, type = "raw")[, "Yes"]

# Ridge and Lasso with repeated cross-validation -----------------------------
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

grid_ridge <- expand.grid(
  alpha = 0,
  lambda = 10^seq(-4, 1, length.out = 30)
)

ridge_fit <- train(
  y ~ ., data = train,
  method = "glmnet",
  family = "binomial",
  metric = "ROC",
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = grid_ridge
)

grid_lasso <- expand.grid(
  alpha = 1,
  lambda = 10^seq(-4, 1, length.out = 30)
)

lasso_fit <- train(
  y ~ ., data = train,
  method = "glmnet",
  family = "binomial",
  metric = "ROC",
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = grid_lasso
)

# ROC / AUC comparison -------------------------------------------------------
auc_logit <- auc(test$y, p_logit)
auc_lda   <- auc(test$y, p_lda)
auc_qda   <- auc(test$y, p_qda)
auc_nb    <- auc(test$y, p_nb)

print(c(
  Logistic = as.numeric(auc_logit),
  LDA = as.numeric(auc_lda),
  QDA = as.numeric(auc_qda),
  NaiveBayes = as.numeric(auc_nb)
))

# The complete coursework also explored SMOTE, GAM/local-regression smoothers
# and polynomial logistic specifications. The central exercise is predictive,
# not causal: model performance is evaluated through classification metrics
# and ROC/AUC rather than interpreted as causal evidence.
