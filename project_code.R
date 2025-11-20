# ------------------------------
# Vincent Johnson
# Joseph Kang
# Catherine Sung
# ------------------------------

# ------------------------------
# Load necessary libraries
# ------------------------------

library(car)
library(caret)
library(cluster)
library(corrplot)
library(dplyr)
library(e1071)
library(factoextra)
library(gbm)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(ISLR2)
library(kableExtra)
library(knitr)
library(leaps)
library(pls)
library(randomForest)
library(tibble)
library(tidyverse)

# ------------------------------
# Read and Merge Data
# ------------------------------

math <- read.csv("math.csv")
por <- read.csv("portuguese.csv")

join_cols <- c("school", "sex", "age", "address", "famsize", "Pstatus",
               "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet")

merged_data <- merge(math, por, by = join_cols, 
                     suffixes = c(".math", ".por"))

# ------------------------------
# Clean and Format Data
# ------------------------------

clean_data <- merged_data %>%
  rename(
    mathp1 = G1.math,
    mathp2 = G2.math,
    mathfinal = G3.math,
    porp1 = G1.por,
    porp2 = G2.por,
    porfinal = G3.por
  ) %>%
  select(starts_with("G"), everything()) %>%
  mutate_if(is.character, as.factor)

clean_data <- clean_data %>%
  select(-ends_with(".math"))

names(clean_data) <- gsub("\\.por$", "", names(clean_data))

# ------------------------------
# Missing Data Check
# ------------------------------

cat("\nMissing data summary:\n")
print(sapply(clean_data, function(x) sum(is.na(x))))

# ------------------------------
# Exploratory Data Analysis (EDA)
# ------------------------------

# Histograms
hist_math <- ggplot(clean_data, aes(x = mathfinal)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Final Math Grades", x = "Grade", y = "Count")

hist_por <- ggplot(clean_data, aes(x = porfinal)) +
  geom_histogram(binwidth = 1, fill = "salmon", color = "black") +
  theme_minimal() +
  labs(title = "Final Portuguese Grades", x = "Grade", y = "Count")

grid.arrange(hist_math, hist_por, ncol = 2)

# Boxplots
box_math <- ggplot(clean_data, aes(x = sex, y = mathfinal)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  labs(title = "Math Grades by Sex", x = "Sex", y = "Math Final Grade")

box_por <- ggplot(clean_data, aes(x = sex, y = porfinal)) +
  geom_boxplot(fill = "lightcoral") +
  theme_minimal() +
  labs(title = "Portuguese Grades by Sex", x = "Sex", y = "Portuguese Final Grade")

grid.arrange(box_math, box_por, ncol = 2)

# QQ Plots
par(mfrow = c(1, 2))
qqnorm(clean_data$mathfinal, main = "QQ Plot - Math")
qqline(clean_data$mathfinal)
qqnorm(clean_data$porfinal, main = "QQ Plot - Portuguese")
qqline(clean_data$porfinal)
par(mfrow = c(1, 1))

# Correlation Heatmap
numeric_data <- clean_data %>% select_if(is.numeric)
cor_matrix <- cor(numeric_data, use = "complete.obs")

corrplot(cor_matrix, method = "color", type = "upper",
         title = "Correlation Matrix (Math & Portuguese)",
         mar = c(0, 0, 1, 0))

# Summary Table
summary_table <- tibble(
  Statistic = c("Min", "1st Quartile", "Median", "Mean",
                "3rd Quartile", "Max", "SD"),
  Math = c(
    min(clean_data$mathfinal),
    quantile(clean_data$mathfinal, 0.25),
    median(clean_data$mathfinal),
    mean(clean_data$mathfinal),
    quantile(clean_data$mathfinal, 0.75),
    max(clean_data$mathfinal),
    sd(clean_data$mathfinal)
  ),
  Portuguese = c(
    min(clean_data$porfinal),
    quantile(clean_data$porfinal, 0.25),
    median(clean_data$porfinal),
    mean(clean_data$porfinal),
    quantile(clean_data$porfinal, 0.75),
    max(clean_data$porfinal),
    sd(clean_data$porfinal)
  )
)

print(summary_table)

kable(summary_table, format = "latex", booktabs = TRUE)

# ------------------------------
# Train/Test Split
# ------------------------------

math_data <- clean_data %>% mutate_if(is.character, as.factor) %>%
  select(-contains("por"))

por_data <- clean_data %>% mutate_if(is.character, as.factor) %>%
  select(-contains("math"))

set.seed(38520251)
train_index <- createDataPartition(math_data$mathfinal, p = 0.7, list = FALSE)
train <- math_data[train_index, ]
test <- math_data[-train_index, ]

set.seed(38520251)
train_indexp <- createDataPartition(por_data$porfinal, p = 0.7, list = FALSE)
trainp <- por_data[train_indexp, ]
testp <- por_data[-train_indexp, ]

# ------------------------------
# Linear Models + Assumption Checks
# ------------------------------

mlr_model <- lm(mathfinal ~ ., data = train)
plr_model <- lm(porfinal ~ ., data = trainp)

summary(mlr_model)
summary(plr_model)

par(mfrow = c(2, 2))
plot(mlr_model)
par(mfrow = c(2, 2))
plot(plr_model)
par(mfrow = c(1, 1))

# VIF
vif_math <- as.data.frame(vif(mlr_model)) %>%
  rownames_to_column(var = "Variable") %>%
  rename_with(~ paste0("Math_", .), -Variable)

vif_por <- as.data.frame(vif(plr_model)) %>%
  rownames_to_column(var = "Variable") %>%
  rename_with(~ paste0("Portuguese_", .), -Variable)

vif_comparison <- full_join(vif_math, vif_por, by = "Variable")

print(vif_comparison)

kable(vif_comparison, format = "latex", booktabs = TRUE)

# ------------------------------
# Lasso Regression
# ------------------------------

# Math
x <- model.matrix(mathfinal ~ ., train)[, -1]
y <- train$mathfinal
lasso_mod <- cv.glmnet(x, y, alpha = 1)
best_lambda <- lasso_mod$lambda.min
lasso_final <- glmnet(x, y, alpha = 1, lambda = best_lambda)

# Portuguese
xp <- model.matrix(porfinal ~ ., trainp)[, -1]
yp <- trainp$porfinal
lasso_modp <- cv.glmnet(xp, yp, alpha = 1)
best_lambdap <- lasso_modp$lambda.min
lasso_finalp <- glmnet(xp, yp, alpha = 1, lambda = best_lambdap)

# Extract coefficients
coef_math <- coef(lasso_final)
coef_por <- coef(lasso_finalp)

coef_math_df <- as.data.frame(as.matrix(coef_math))
coef_math_df <- tibble::rownames_to_column(coef_math_df, var = "Variable")
colnames(coef_math_df)[2] <- "Math_Coefficient"

coef_por_df <- as.data.frame(as.matrix(coef_por))
coef_por_df <- tibble::rownames_to_column(coef_por_df, var = "Variable")
colnames(coef_por_df)[2] <- "Portuguese_Coefficient"

lasso_compare <- full_join(coef_math_df, coef_por_df, by = "Variable")

lasso_compare_filtered <- lasso_compare %>%
  filter(Math_Coefficient != 0 | Portuguese_Coefficient != 0)

kable(lasso_compare_filtered, format = "latex", booktabs = TRUE)

# ------------------------------
# Bagging & Boosting
# ------------------------------

set.seed(38520251)
bag_mod <- randomForest(mathfinal ~ ., data = train, mtry = ncol(train) - 1)
boost_mod <- gbm(mathfinal ~ ., data = train,
                 distribution = "gaussian", n.trees = 1000, interaction.depth = 4)

set.seed(38520251)
bag_modp <- randomForest(porfinal ~ ., data = trainp, mtry = ncol(trainp) - 1)
boost_modp <- gbm(porfinal ~ ., data = trainp,
                  distribution = "gaussian", n.trees = 1000, interaction.depth = 4)

pred_bag <- predict(bag_mod, newdata = test)
pred_boost <- predict(boost_mod, newdata = test, n.trees = 1000)

pred_bagp <- predict(bag_modp, newdata = testp)
pred_boostp <- predict(boost_modp, newdata = testp, n.trees = 1000)

rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))

rmse_bag_math <- rmse(test$mathfinal, pred_bag)
rmse_boost_math <- rmse(test$mathfinal, pred_boost)

rmse_bag_por <- rmse(testp$porfinal, pred_bagp)
rmse_boost_por <- rmse(testp$porfinal, pred_boostp)

# ------------------------------
# PCR & PLS
# ------------------------------

pcr_mod <- pcr(mathfinal ~ ., data = train, scale = TRUE, validation = "CV")
pcr_modp <- pcr(porfinal ~ ., data = trainp, scale = TRUE, validation = "CV")

pls_mod <- plsr(mathfinal ~ ., data = train, scale = TRUE, validation = "CV")
pls_modp <- plsr(porfinal ~ ., data = trainp, scale = TRUE, validation = "CV")

opt_comp_pcr_math <- which.min(MSEP(pcr_mod)$val[1, , ])
opt_comp_pls_math <- which.min(MSEP(pls_mod)$val[1, , ])

opt_comp_pcr_por <- which.min(MSEP(pcr_modp)$val[1, , ])
opt_comp_pls_por <- which.min(MSEP(pls_modp)$val[1, , ])

# Predictions
pred_pcr <- predict(pcr_mod, newdata = test, ncomp = opt_comp_pcr_math)
pred_pls <- predict(pls_mod, newdata = test, ncomp = opt_comp_pls_math)

pred_pcrp <- predict(pcr_modp, newdata = testp, ncomp = opt_comp_pcr_por)
pred_plsp <- predict(pls_modp, newdata = testp, ncomp = opt_comp_pls_por)

# RMSE
rmse_pcr_math <- rmse(test$mathfinal, pred_pcr)
rmse_pls_math <- rmse(test$mathfinal, pred_pls)

rmse_pcr_por <- rmse(testp$porfinal, pred_pcrp)
rmse_pls_por <- rmse(testp$porfinal, pred_plsp)

# ------------------------------
# Random Forest & SVM
# ------------------------------

rf_mod <- randomForest(mathfinal ~ ., data = train, importance = TRUE)
rf_modp <- randomForest(porfinal ~ ., data = trainp, importance = TRUE)

pred_rf <- predict(rf_mod, newdata = test)
pred_rfp <- predict(rf_modp, newdata = testp)

svm_mod <- svm(mathfinal ~ ., data = train)
svm_modp <- svm(porfinal ~ ., data = trainp)

pred_svm <- predict(svm_mod, newdata = test)
pred_svmp <- predict(svm_modp, newdata = testp)

# RMSE
rmse_rf_math <- rmse(test$mathfinal, pred_rf)
rmse_svm_math <- rmse(test$mathfinal, pred_svm)

rmse_rf_por <- rmse(testp$porfinal, pred_rfp)
rmse_svm_por <- rmse(testp$porfinal, pred_svmp)

# ------------------------------
# Final Output Table
# ------------------------------

final_metrics <- data.frame(
  Model = c("Bagging_Math", "Boosting_Math", "RF_Math", "SVM_Math",
            "PCR_Math", "PLS_Math",
            "Bagging_Por", "Boosting_Por", "RF_Por", "SVM_Por",
            "PCR_Por", "PLS_Por"),
  RMSE = c(
    rmse_bag_math, rmse_boost_math, rmse_rf_math, rmse_svm_math,
    rmse_pcr_math, rmse_pls_math,
    rmse_bag_por, rmse_boost_por, rmse_rf_por, rmse_svm_por,
    rmse_pcr_por, rmse_pls_por
  )
)

kable(final_metrics, format = "latex", booktabs = TRUE)
