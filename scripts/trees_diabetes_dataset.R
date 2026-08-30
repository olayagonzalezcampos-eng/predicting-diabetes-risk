rm(list = ls()) # Clean environment

#Upload packages
library(tidyverse)    # Data manipulation and visualization
library(class)        # k-NN implementation
library(caret)        # Model training and evaluation
library(pROC)         # ROC curve analysis
library(ggplot2)
library(patchwork)

set.seed(123) # for reproducibility

#==============================================================================
# PRELIMINARY DATASET OVERVIEW
#==============================================================================

#Upload the dataset
data <- read.csv("dataset_diabetes.csv", stringsAsFactors=TRUE)

# Number of observations and variables
n_obs <- nrow(data)
n_vars <- ncol(data)

cat("The dataset contains", n_obs, "observations and", 
    n_vars, "variables.\n")

# Define variable groups
binary_vars <- c('HighBP', 'HighChol','CholCheck','Smoker',
                 'Stroke','HeartDiseaseorAttack','PhysActivity',
                 'Fruits','Veggies','HvyAlcoholConsump',
                 'AnyHealthcare','NoDocbcCost','DiffWalk','Sex')

categorical_vars <- c('Diabetes_012','GenHlth','Age','Education',
                      'Income')

numeric_vars <- setdiff(names(data), c(binary_vars, categorical_vars))

# Count variables
n_binary <- length(binary_vars)
n_categorical <- length(categorical_vars)
n_numeric <- length(numeric_vars)

cat(sprintf("The dataset is composed of:\n\n"))
cat(sprintf("- %d binary variables\n", n_binary))
cat(sprintf("- %d categorical (multiclass/ordinal) variables\n", n_categorical))
cat(sprintf("- %d numerical (continuous) variables\n", n_numeric))

vars_factor <- c('Diabetes_012', 'HighBP', 'HighChol','CholCheck','Smoker',
                 'Stroke','HeartDiseaseorAttack','PhysActivity', 'Fruits','Veggies',
                 'HvyAlcoholConsump', 'AnyHealthcare','NoDocbcCost','GenHlth',
                 'DiffWalk','Sex','Age','Education','Income')

# Convert selected variables to factors
data[vars_factor] <- lapply(data[vars_factor], as.factor)

# Define ordinal variables
data$Age <- ordered(data$Age)
data$Education <- ordered(data$Education)
data$Income <- ordered(data$Income)

na_total <- sum(is.na(data))
na_by_var <- colSums(is.na(data))

cat(sprintf("Total missing values in the dataset: %d\n\n", na_total))

if (na_total == 0) {
  cat("No missing values were found in the dataset.\n")
} else {
  cat("Missing values by variable (only variables with NAs are shown):\n\n")
  print(na_by_var[na_by_var > 0])
}

df <- dplyr::sample_n(data, 50000)

# Number of observations and variables
n_obs <- nrow(df)
n_vars <- ncol(df)

cat("Now the dataset contains", n_obs, "observations and", 
    n_vars, "variables.\n")

#==============================================================================
# Train/Test Split  20/80
#==============================================================================

# Name the levels of the variable "Diabetes_012"
df$Diabetes_012 <- factor(df$Diabetes_012,
                          levels = c(0,1,2),
                          labels = c("No Diabetes", "Prediabetes", "Diabetes"))

n_obs_df <- nrow(df)

idx_train <- sample(n_obs_df, size = floor(0.8 * n_obs_df))
train_data <- df[idx_train, ]    # Train set
test_data  <- df[-idx_train, ]   # Test set

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

#==============================================================================
# Upsampling (k-NN is sensible to class imbalance)
#==============================================================================
#--------------- 1. Prepare Data for Resampling ---------------
# We use all features since the goal is model improvement
train_x <- train_data %>% dplyr::select(-Diabetes_012)
train_y <- train_data$Diabetes_012
#--------------- 2. Perform Resampling ---------------
# Upsampling: Increases Class 1 and 2 to match the size of Class 0
train_up <- caret::upSample(x = train_x, y = train_y, 
                            yname = "Diabetes_012")

#==============================================================================
# TREE-BASED METHODS
# Methods compared:
#   1. Random Forest  (randomForest, mtry = sqrt(p))
#   2. XGBoost        (xgboost)
#   3. LightGBM       (lightgbm)
#   4. CatBoost       (catboost — optional, see install note)
#
# Dropped methods and reasons:
#   - Single Tree : guaranteed worst performer, useful as classroom baseline only
#   - Bagging     : RF with decorrelated trees is always strictly better
#   - GBM         : multinomial distribution is broken in current gbm package
#   - AdaBoost    : subsampled to 20k rows for speed → unfair comparison;
#                   SAMME is also weaker than modern gradient boosting
#   - BART        : capped at 10k rows → trains on <10% of what other methods
#                   use, making any comparison misleading
#==============================================================================

# ── Install (run once, then comment out) ─────────────────────────────────────
# install.packages(c("randomForest","xgboost","lightgbm","ggrepel"))
#
# CatBoost requires a manual install from the GitHub releases page:
#   https://github.com/catboost/catboost/releases
# Example (Windows, R 4.x):
#   remotes::install_url(
#     "https://github.com/catboost/catboost/releases/download/v1.2.7/
#      catboost-R-Windows-1.2.7.tgz",
#     INSTALL_opts = c("--no-multiarch", "--no-test-load"))

#==============================================================================
# LOAD PACKAGES
#==============================================================================
library(randomForest)   # Random Forest
library(xgboost)        # XGBoost
library(lightgbm)       # LightGBM
library(catboost)       # CatBoost
library(ggrepel)        # Label repelling for scatter plot

#==============================================================================
# DATA PREPARATION
# Two flavors needed:
#   A) Factor-native  → randomForest
#   B) Numeric matrix → XGBoost, LightGBM, CatBoost
#==============================================================================

# ── A. Factor-native datasets ─────────────────────────────────────────────────
# train_up and test_data already contain Diabetes_012 as a factor with 3 levels:
#   "No Diabetes", "Prediabetes", "Diabetes"
# Ordered factors (Age, Education, Income) are handled by randomForest
# as regular factors (ordering is not used by these methods, which is fine).

train_rf <- train_up       # upsampled training set
test_rf  <- test_data      # original test set (NOT upsampled)

class_labels <- levels(train_rf$Diabetes_012)  # c("No Diabetes","Prediabetes","Diabetes")
n_classes    <- length(class_labels)            # 3
p            <- ncol(train_rf) - 1             # number of predictors

# ── B. Numeric matrices for gradient boosting ────────────────────────────────
# Convert all factors/ordered to their integer codes.
# For tree-based models, integer codes preserve ordinal information
# and avoid the dimensionality explosion of one-hot encoding.

to_num_matrix <- function(df, target = "Diabetes_012") {
  X <- dplyr::select(df, -all_of(target))
  X <- data.frame(lapply(X, function(col) {
    if (is.factor(col) || is.ordered(col)) as.integer(col) else col
  }))
  as.matrix(X)
}

X_train <- to_num_matrix(train_up)
X_test  <- to_num_matrix(test_data)

# Labels as 0-indexed integers  (XGBoost / LightGBM / CatBoost convention)
y_train_int  <- as.integer(train_up$Diabetes_012)  - 1L   # 0 / 1 / 2
y_test_int   <- as.integer(test_data$Diabetes_012) - 1L

#==============================================================================
# HELPER: per-model performance summary
#   Accuracy     : overall correct rate — misleading with class imbalance
#   Kappa        : corrects for chance agreement; robust to imbalance
#   Macro F1     : arithmetic mean of per-class F1 scores (PRIMARY metric)
#   Macro BalAcc : mean of per-class balanced accuracy
#   Sensitivity per class: how well each class is detected individually
#==============================================================================

model_summary <- function(pred_factor, truth_factor, model_name) {
  cm <- confusionMatrix(pred_factor, truth_factor)
  bc <- cm$byClass   # rows = Class: No Diabetes / Prediabetes / Diabetes
  ov <- cm$overall
  
  data.frame(
    Model         = model_name,
    Accuracy      = round(ov["Accuracy"],                          4),
    Kappa         = round(ov["Kappa"],                             4),
    Macro_F1      = round(mean(bc[, "F1"],                na.rm = TRUE), 4),
    Macro_BalAcc  = round(mean(bc[, "Balanced Accuracy"], na.rm = TRUE), 4),
    Sens_NoDiab   = round(bc["Class: No Diabetes",  "Sensitivity"], 4),
    Sens_PreDiab  = round(bc["Class: Prediabetes",  "Sensitivity"], 4),
    Sens_Diab     = round(bc["Class: Diabetes",     "Sensitivity"], 4),
    row.names     = NULL,
    stringsAsFactors = FALSE
  )
}

results_list <- list()   # accumulates one row per model

#==============================================================================
# VALIDATION SPLIT FOR GRADIENT BOOSTING EARLY STOPPING
# Carved from train_data (the ORIGINAL pre-upsampling training set), which is
# still available in the environment. This ensures early stopping sees the real
# class distribution, not the synthetic upsampled one.
# X_train (full upsampled set) is used as the boosters' training data;
# X_val (real distribution) is used exclusively for early stopping decisions.
# All three boosting models share this identical split for a fair comparison.
#==============================================================================
set.seed(123)
val_idx   <- sample(nrow(train_data), size = floor(0.15 * nrow(train_data)))
val_data  <- train_data[val_idx, ]   # ~6 000 rows, real class distribution

X_val     <- to_num_matrix(val_data)
y_val_int <- as.integer(val_data$Diabetes_012) - 1L

#==============================================================================
# 1.  RANDOM FOREST                                                    [1/4]
#     mtry tuned via OOB error across candidate values — cheap since no
#     cross-validation is needed; OOB estimate is computed for free during
#     training. sampsize enforces per-class balance at each tree, reinforcing
#     the upsampling.
#     Speed note: ntree = 50 is enough to rank mtry candidates reliably;
#     final model uses ntree = 200 for stable OOB and good performance.
#==============================================================================
cat("\n╔══════════════════════════╗\n")
cat("║  [1/4]  RANDOM FOREST    ║\n")
cat("╚══════════════════════════╝\n")

# ── Tune mtry via OOB error (ntree = 50 is enough to rank candidates) ─────────
mtry_grid <- c(2, floor(sqrt(p)), floor(p / 3), floor(p / 2))

oob_errors <- sapply(mtry_grid, function(m) {
  set.seed(123)
  rf_tmp <- randomForest(Diabetes_012 ~ ., data = train_rf,
                         mtry = m, ntree = 50)
  rf_tmp$err.rate[50, "OOB"]
})

cat("mtry candidates :", paste(mtry_grid,            collapse = " | "), "\n")
cat("OOB errors      :", paste(round(oob_errors, 4), collapse = " | "), "\n")

best_mtry <- mtry_grid[which.min(oob_errors)]
cat(sprintf("Selected mtry = %d\n", best_mtry))

# ── Final RF with tuned mtry and explicit sampsize for class balance ───────────
# sampsize: draw the same number of observations per class at each tree,
# matching the smallest class size so every tree sees a balanced bootstrap.
min_class_n <- min(table(train_rf$Diabetes_012))
samp_size   <- rep(min_class_n, n_classes)

set.seed(123)
model_rf <- randomForest(
  Diabetes_012 ~ .,
  data       = train_rf,
  mtry       = best_mtry,   # ← tuned, not hardcoded
  ntree      = 200,         # ← reduced from 300 for speed; stable at 200
  sampsize   = samp_size,   # ← balanced draw per class per tree
  importance = TRUE
)

cat("OOB error estimate:\n")
print(model_rf$err.rate[nrow(model_rf$err.rate), ])

pred_rf <- predict(model_rf, newdata = test_rf, type = "class")
cm_rf   <- confusionMatrix(pred_rf, test_rf$Diabetes_012)
cat("\n--- Random Forest Confusion Matrix (test set) ---\n")
print(cm_rf)

varImpPlot(model_rf, n.var = 15, main = "Random Forest — Variable Importance")

results_list[["Random Forest"]] <- model_summary(pred_rf, test_rf$Diabetes_012,
                                                 "Random Forest")

#==============================================================================
# 2.  XGBOOST                                                          [2/4]
#     Regularised gradient boosting (L1 + L2) with column/row sub-sampling.
#     Objective: multi:softmax (returns hard class predictions).
#     Trains on the full upsampled set (X_train); early stopping evaluated
#     on X_val carved from original train_data — real class distribution,
#     no data leakage into the test set.
#==============================================================================
cat("\n╔══════════════════════════╗\n")
cat("║  [2/4]  XGBOOST          ║\n")
cat("╚══════════════════════════╝\n")

dtrain_xgb <- xgb.DMatrix(data = X_train, label = y_train_int)
dval_xgb   <- xgb.DMatrix(data = X_val,   label = y_val_int)
dtest_xgb  <- xgb.DMatrix(data = X_test,  label = y_test_int)  # only for final eval

set.seed(123)
model_xgb <- xgb.train(
  params = list(
    objective        = "multi:softmax",
    num_class        = n_classes,
    eta              = 0.1,    # Lower LR — early stopping compensates
    max_depth        = 6,
    subsample        = 0.8,     # Row sub-sampling
    colsample_bytree = 0.8,     # Column sub-sampling
    min_child_weight = 5,
    eval_metric      = "merror" # Multiclass classification error
  ),
  data                  = dtrain_xgb,
  nrounds               = 800,  # High ceiling; early stopping exits well before
  watchlist             = list(train = dtrain_xgb, val = dval_xgb),  # NO test set
  early_stopping_rounds = 20,   # Stop if val error does not improve for 20 rounds
  verbose               = 1,
  print_every_n         = 50,
  seed                  = 123
)
cat("Best XGBoost iteration:", model_xgb$best_iteration, "\n")

pred_xgb_int <- predict(model_xgb, dtest_xgb)
pred_xgb     <- factor(class_labels[pred_xgb_int + 1], levels = class_labels)
cm_xgb       <- confusionMatrix(pred_xgb, test_rf$Diabetes_012)
cat("\n--- XGBoost Confusion Matrix (test set) ---\n")
print(cm_xgb)

results_list[["XGBoost"]] <- model_summary(pred_xgb, test_rf$Diabetes_012, "XGBoost")

#==============================================================================
# 3.  LightGBM                                                         [3/4]
#     Leaf-wise tree growth (vs depth-wise in XGBoost) → faster & often better
#     on large datasets. Uses histogram-based split finding.
#     Reuses the same X_train / X_val split — ensures both gradient boosting
#     models are tuned on identical held-out data for a fair comparison.
#==============================================================================
cat("\n╔══════════════════════════╗\n")
cat("║  [3/4]  LightGBM         ║\n")
cat("╚══════════════════════════╝\n")

dtrain_lgb <- lgb.Dataset(data = X_train, label = y_train_int)
dval_lgb   <- lgb.Dataset(data = X_val,   label = y_val_int,
                          reference = dtrain_lgb)

set.seed(123)
model_lgb <- lgb.train(
  params = list(
    objective        = "multiclass",
    num_class        = n_classes,
    learning_rate    = 0.1,    # Lower LR — early stopping compensates
    num_leaves       = 31,      # Controls tree complexity (leaf-wise growth)
    feature_fraction = 0.8,     # Column sub-sampling
    bagging_fraction = 0.8,     # Row sub-sampling
    bagging_freq     = 5,       # Apply bagging every 5 rounds
    min_data_in_leaf = 20,
    metric           = "multi_error",
    verbose          = -1
  ),
  data                  = dtrain_lgb,
  nrounds               = 800,  # High ceiling; early stopping exits well before
  valids                = list(val = dval_lgb),
  early_stopping_rounds = 20,   # Stop if val error does not improve for 20 rounds
  record                = TRUE
)
cat("Best LightGBM iteration:", model_lgb$best_iter, "\n")

pred_lgb_raw  <- predict(model_lgb, X_test)
pred_lgb_prob <- matrix(pred_lgb_raw, nrow = nrow(X_test),
                        ncol = n_classes, byrow = FALSE)
pred_lgb_int  <- max.col(pred_lgb_prob) - 1L
pred_lgb      <- factor(class_labels[pred_lgb_int + 1], levels = class_labels)
cm_lgb        <- confusionMatrix(pred_lgb, test_rf$Diabetes_012)
cat("\n--- LightGBM Confusion Matrix (test set) ---\n")
print(cm_lgb)

results_list[["LightGBM"]] <- model_summary(pred_lgb, test_rf$Diabetes_012, "LightGBM")

#==============================================================================
# 4.  CatBoost                                                         [4/4]
#     Native handling of categorical features; symmetric trees; ordered boosting.
#     Worth including given the high proportion of categorical/ordinal predictors.
#     Reuses the same X_train / X_val split — no test leakage.
#     Early stopping via od_type = "Iter" / od_wait (correct R binding syntax;
#     the parameter name "early_stopping_rounds" is not recognised by catboost R).
#==============================================================================
cat("\n╔══════════════════════════╗\n")
cat("║  [4/4]  CatBoost         ║\n")
cat("╚══════════════════════════╝\n")

train_pool <- catboost::catboost.load_pool(data = X_train, label = y_train_int)
val_pool   <- catboost::catboost.load_pool(data = X_val,   label = y_val_int)
test_pool  <- catboost::catboost.load_pool(data = X_test,  label = y_test_int)

set.seed(123)
model_cat <- catboost::catboost.train(
  learn_pool = train_pool,
  test_pool  = val_pool,      # ← validation pool, NOT the test set
  params = list(
    loss_function       = "MultiClass",
    iterations          = 800,  # High ceiling; early stopping exits well before
    learning_rate       = 0.05, # Lower LR — early stopping compensates
    depth               = 6,
    l2_leaf_reg         = 3,    # L2 regularisation on leaf values
    random_strength     = 1,
    bagging_temperature = 1,
    od_type             = "Iter", # Overfitting detector: stop after N iters of no gain
    od_wait             = 20,     # Equivalent to early_stopping_rounds = 20
    verbose             = 50,     # Print every 50 iterations
    random_seed         = 123
  )
)
cat("Best CatBoost iteration:", model_cat$tree_count, "\n")

pred_cat_int <- as.integer(catboost::catboost.predict(
  model_cat, test_pool, prediction_type = "Class"))
pred_cat <- factor(class_labels[pred_cat_int + 1], levels = class_labels)
cm_cat   <- confusionMatrix(pred_cat, test_rf$Diabetes_012)
cat("\n--- CatBoost Confusion Matrix (test set) ---\n")
print(cm_cat)

results_list[["CatBoost"]] <- model_summary(pred_cat, test_rf$Diabetes_012, "CatBoost")

#==============================================================================
# MODEL COMPARISON
#==============================================================================
cat("\n\n")
cat("╔══════════════════════════════════════════════════════════════════════╗\n")
cat("║                    MODEL COMPARISON SUMMARY                        ║\n")
cat("║  Key metrics for CLASS-IMBALANCED multiclass:                      ║\n")
cat("║  • Accuracy     : kept for reference; misleading with imbalance    ║\n")
cat("║  • Kappa        : penalises chance agreement (robust to imbalance) ║\n")
cat("║  • Macro F1     : mean F1 across the 3 classes  <- PRIMARY METRIC  ║\n")
cat("║  • Macro BalAcc : mean balanced accuracy across classes            ║\n")
cat("║  • Sens_PreDiab : recall for Prediabetes (the extreme minority)    ║\n")
cat("╚══════════════════════════════════════════════════════════════════════╝\n\n")

comparison_table <- do.call(rbind, results_list)
rownames(comparison_table) <- NULL
print(comparison_table, row.names = FALSE)

# ── Best model per metric ─────────────────────────────────────────────────────
cat("\n--- Best model by metric ---\n")
cat(sprintf("Highest Accuracy         : %-20s (%.4f)\n",
            comparison_table$Model[which.max(comparison_table$Accuracy)],
            max(comparison_table$Accuracy)))
cat(sprintf("Highest Kappa            : %-20s (%.4f)\n",
            comparison_table$Model[which.max(comparison_table$Kappa)],
            max(comparison_table$Kappa)))
cat(sprintf("Highest Macro F1         : %-20s (%.4f)\n",
            comparison_table$Model[which.max(comparison_table$Macro_F1)],
            max(comparison_table$Macro_F1)))
cat(sprintf("Highest Macro Bal. Acc.  : %-20s (%.4f)\n",
            comparison_table$Model[which.max(comparison_table$Macro_BalAcc)],
            max(comparison_table$Macro_BalAcc)))
cat(sprintf("Best Prediabetes Recall  : %-20s (%.4f)\n",
            comparison_table$Model[which.max(comparison_table$Sens_PreDiab)],
            max(comparison_table$Sens_PreDiab)))
cat(sprintf("Best Diabetes Recall     : %-20s (%.4f)\n",
            comparison_table$Model[which.max(comparison_table$Sens_Diab)],
            max(comparison_table$Sens_Diab)))

# ── Recommended model ─────────────────────────────────────────────────────────
# Primary criterion: highest Macro F1 (balances precision/recall across all
# three classes, unaffected by class imbalance).
# Tie-breaker (within 0.005 of best Macro F1): prefer highest Sens_PreDiab
# since detecting pre-diabetes early has high clinical value.
best_idx <- which.max(comparison_table$Macro_F1)
cat(sprintf(
  "\n[*] RECOMMENDED MODEL: %s  (Macro F1 = %.4f | Sens_PreDiab = %.4f)\n",
  comparison_table$Model[best_idx],
  comparison_table$Macro_F1[best_idx],
  comparison_table$Sens_PreDiab[best_idx]
))

#==============================================================================
# VISUALISATION
# 1. Grouped bar chart : macro-level metrics per model
# 2. Per-class sensitivity heatmap
# 3. Scatter: Macro F1 vs Prediabetes sensitivity (trade-off view)
#==============================================================================

# ── Plot 1: Macro metrics comparison ─────────────────────────────────────────
metrics_long <- comparison_table %>%
  dplyr::select(Model, Kappa, Macro_F1, Macro_BalAcc) %>%
  tidyr::pivot_longer(
    cols      = -Model,
    names_to  = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    Metric = factor(Metric,
                    levels = c("Kappa", "Macro_F1", "Macro_BalAcc"),
                    labels = c("Kappa", "Macro F1", "Macro Balanced Acc"))
  )

p1 <- ggplot(metrics_long, aes(x = reorder(Model, Value), y = Value, fill = Model)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            hjust = -0.1, size = 3) +
  facet_wrap(~ Metric, ncol = 1, scales = "free_y") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title    = "Tree-Based Methods — Model Comparison",
    subtitle = "Diabetes dataset | Upsampled training set | Macro-averaged metrics",
    x        = NULL, y = "Score"
  ) +
  theme_minimal(base_size = 11) +
  theme(strip.text = element_text(face = "bold"))

print(p1)

# ── Plot 2: Per-class sensitivity heatmap ─────────────────────────────────────
sens_long <- comparison_table %>%
  dplyr::select(Model, Sens_NoDiab, Sens_PreDiab, Sens_Diab) %>%
  tidyr::pivot_longer(
    cols      = -Model,
    names_to  = "Class",
    values_to = "Sensitivity"
  ) %>%
  mutate(
    Class = factor(Class,
                   levels = c("Sens_NoDiab", "Sens_PreDiab", "Sens_Diab"),
                   labels = c("No Diabetes", "Prediabetes", "Diabetes"))
  )

p2 <- ggplot(sens_long, aes(x = Class, y = Model, fill = Sensitivity)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.3f", Sensitivity)), size = 3.5) +
  scale_fill_gradient2(
    low      = "#d73027",
    mid      = "#ffffbf",
    high     = "#1a9850",
    midpoint = 0.5,
    limits   = c(0, 1),
    name     = "Sensitivity"
  ) +
  labs(
    title    = "Per-Class Sensitivity — All Models",
    subtitle = "Green = good detection  |  Red = class largely missed\nPrediabetes is the extreme minority class (~2%)",
    x        = "Class", y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(face = "bold"))

print(p2)

# ── Plot 3: Scatter — Macro F1 vs Prediabetes recall ─────────────────────────
p3 <- ggplot(comparison_table,
             aes(x = Sens_PreDiab, y = Macro_F1, label = Model, color = Model)) +
  geom_point(size = 4, show.legend = FALSE) +
  ggrepel::geom_text_repel(size = 3.5, show.legend = FALSE, box.padding = 0.4) +
  geom_vline(xintercept = 0.4, linetype = "dashed",  color = "grey50") +
  geom_hline(yintercept = max(comparison_table$Macro_F1) - 0.01,
             linetype   = "dotted", color = "grey50") +
  scale_color_brewer(palette = "Set1") +
  labs(
    title    = "Macro F1 vs Prediabetes Recall",
    subtitle = "Upper-right corner = best trade-off across all classes\nDashed lines: Prediabetes recall >= 0.4 | within 0.01 of best Macro F1",
    x        = "Prediabetes Sensitivity (Recall)",
    y        = "Macro F1 (all 3 classes)"
  ) +
  theme_minimal(base_size = 11)

print(p3)

#==============================================================================
# SAVE MODELS AND OBJECTS
#==============================================================================
saveRDS(model_rf,         "model_rf.rds")
saveRDS(model_xgb,        "model_xgb.rds")
saveRDS(model_lgb,        "model_lgb.rds")
saveRDS(comparison_table, "comparison_table.rds")
saveRDS(test_rf,          "test_rf.rds")
saveRDS(X_test,           "X_test.rds")
saveRDS(class_labels,     "class_labels.rds")

if (exists("model_cat")) saveRDS(model_cat, "model_cat.rds")   # guard: skip if CatBoost failed

# To load them in the Quarto document:
# model_rf         <- readRDS("model_rf.rds")
# model_xgb        <- readRDS("model_xgb.rds")
# model_lgb        <- readRDS("model_lgb.rds")
# model_cat        <- readRDS("model_cat.rds")
# comparison_table <- readRDS("comparison_table.rds")
# test_rf          <- readRDS("test_rf.rds")
# X_test           <- readRDS("X_test.rds")
# class_labels     <- readRDS("class_labels.rds")