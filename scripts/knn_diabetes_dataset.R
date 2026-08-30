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
# KNN for classification
#==============================================================================

# Pairwise scatter plots
ggplot(df, aes(x = PhysHlth, y = MentHlth, color = Diabetes_012)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Diabetes Dataset",
       x = "Physical Health", y = "Mental Health") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

# Scaling the numerical variables
#---------------------------------------------------------
# Without scaling - distances dominated by large features
cat("Range before scaling:\n")
sapply(train_up[,numeric_vars], range)

# With scaling
train_means <- colMeans(train_up[, numeric_vars])
train_sds   <- apply(train_up[, numeric_vars], 2, sd)

train_scaled <- train_up
train_scaled[, numeric_vars] <- scale(train_up[, numeric_vars],
                                      center = train_means,
                                      scale  = train_sds)

test_scaled <- test_data
test_scaled[, numeric_vars] <- scale(test_data[, numeric_vars],
                                     center = train_means,
                                     scale  = train_sds)

cat("\nRange after scaling:\n")
sapply(as.data.frame(train_scaled[,numeric_vars]), range)

#---------------------------------------------------------
# Using k-NN and decision boundaries
#---------------------------------------------------------

# k-NN with k = 3

train_matrix <- as.matrix(train_scaled[, numeric_vars])
test_matrix  <- as.matrix(test_scaled[, numeric_vars])

train_matrix <- train_matrix + matrix(rnorm(length(train_matrix), 0, 1e-8),
                                      nrow = nrow(train_matrix))

test_matrix  <- test_matrix + matrix(rnorm(length(test_matrix), 0, 1e-8),
                                     nrow = nrow(test_matrix))

knn_pred_k3 <- knn(train = train_matrix,
                   test  = test_matrix,
                   cl    = train_up$Diabetes_012,
                   k     = 3)

# Confusion matrix
confusionMatrix(knn_pred_k3, test_data$Diabetes_012)

# Visualization of the decision boundaries
#---------------------------------------------------------

numeric_vars_vis <- c("MentHlth", "PhysHlth")

train_means_vis <- colMeans(train_up[, numeric_vars_vis])
train_sds_vis   <- apply(train_up[, numeric_vars_vis], 2, sd)

# We keep this scaled version for the KNN algorithm to use (Upsampled)
train_scaled_vis <- train_up
train_scaled_vis[, numeric_vars_vis] <- scale(train_up[, numeric_vars_vis],
                                              center = train_means_vis,
                                              scale  = train_sds_vis)

# Prepare a separate dataset for plotting points based on ORIGINAL train_data
# This prevents the "overcrowding" caused by upsampling in the visual layer
plot_points <- train_data
plot_points[, numeric_vars_vis] <- scale(train_data[, numeric_vars_vis],
                                         center = train_means_vis,
                                         scale  = train_sds_vis)

# Add jitter to the points so we can see the density of the overlapping data
set.seed(123)
plot_points$MentHlth <- plot_points$MentHlth + rnorm(nrow(plot_points), 0, 0.05)
plot_points$PhysHlth <- plot_points$PhysHlth + rnorm(nrow(plot_points), 0, 0.05)

# train_vis is used for the KNN training inside the grid calculation
train_vis <- train_scaled_vis
train_vis[, numeric_vars_vis] <-
  train_vis[, numeric_vars_vis] +
  matrix(rnorm(nrow(train_vis) * 2, 0, 1e-8), ncol = 2)

x_range <- seq(min(plot_points[, "MentHlth"]) - 0.5,
               max(plot_points[, "MentHlth"]) + 0.5,
               length.out = 100)

y_range <- seq(min(plot_points[, "PhysHlth"]) - 0.5,
               max(plot_points[, "PhysHlth"]) + 0.5,
               length.out = 100)

grid <- expand.grid(MentHlth = x_range,
                    PhysHlth = y_range)

grid$k1 <- knn(train = train_vis[, numeric_vars_vis],
               test  = as.matrix(grid[, numeric_vars_vis]),
               cl    = train_up$Diabetes_012,
               k     = 1)

grid$k5 <- knn(train = train_vis[, numeric_vars_vis],
               test  = as.matrix(grid[, numeric_vars_vis]),
               cl    = train_up$Diabetes_012,
               k     = 5)

grid$k15 <- knn(train = train_vis[, numeric_vars_vis],
                test  = as.matrix(grid[, numeric_vars_vis]),
                cl    = train_up$Diabetes_012,
                k     = 15)

# UPDATED: We use the plot_points (Original Data) instead of train_up for geom_point
plot_data <- plot_points 

p_k1 <- ggplot() +
  geom_tile(data = grid,
            aes(x = MentHlth, y = PhysHlth, fill = k1),
            alpha = 0.3) +
  geom_point(data = plot_data,
             aes(x = MentHlth, y = PhysHlth, color = Diabetes_012),
             size = 1.2, alpha = 0.4) + # Reduced size and added alpha to see density
  labs(title = "k = 1 (High Variance)") +
  theme_minimal() +
  theme(legend.position = "none")

p_k5 <- ggplot() +
  geom_tile(data = grid,
            aes(x = MentHlth, y = PhysHlth, fill = k5),
            alpha = 0.3) +
  geom_point(data = plot_data,
             aes(x = MentHlth, y = PhysHlth, color = Diabetes_012),
             size = 1.2, alpha = 0.4) +
  labs(title = "k = 5 (Balanced)") +
  theme_minimal() +
  theme(legend.position = "none")

p_k15 <- ggplot() +
  geom_tile(data = grid,
            aes(x = MentHlth, y = PhysHlth, fill = k15),
            alpha = 0.3) +
  geom_point(data = plot_data,
             aes(x = MentHlth, y = PhysHlth, color = Diabetes_012),
             size = 1.2, alpha = 0.4) +
  labs(title = "k = 15 (High Bias)") +
  theme_minimal() +
  theme(legend.position = "bottom")


# Combine plots
(p_k1 | p_k5 | p_k15) +
  plot_annotation(
    title = "k-NN Decision Boundaries: Effect of k",
    subtitle = "Background: Decision zones (Upsampled) | Points: Actual distribution (Jittered)"
  )

#---------------------------------------------------------
# Selecting the optimal k
#---------------------------------------------------------

k_values <- 1:15
kappa_values <- numeric(length(k_values)) 

for (i in seq_along(k_values)) {
  cat("Running k =", k_values[i], "...\n")
  
  pred <- knn(train = train_matrix,
              test  = test_matrix,
              cl    = train_up$Diabetes_012,
              k     = k_values[i])
  
  # Calculate Kappa using confusionMatrix
  cm <- confusionMatrix(pred, test_data$Diabetes_012)
  kappa_values[i] <- cm$overall['Kappa']
}

k_results <- data.frame(k = k_values, Kappa = kappa_values)

ggplot(k_results, aes(x = k, y = Kappa)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "steelblue", size = 2) +
  geom_vline(xintercept = k_values[which.max(kappa_values)],
             linetype = "dashed", color = "red") +
  # Changed + 2 to - 0.5 and added hjust to align text to the right
  annotate("text", 
           x = k_values[which.max(kappa_values)] - 0.5, 
           y = max(kappa_values),
           label = paste("Optimal k =", k_values[which.max(kappa_values)]),
           color = "red",
           hjust = 1) + 
  labs(title = "k-NN: Selecting Optimal k (Based on Kappa)",
       subtitle = "Test Kappa statistic for different values of k",
       x = "Number of Neighbors (k)", y = "Kappa") +
  theme_minimal()

cat("Optimal k:", k_values[which.max(kappa_values)], "\n")
cat("Best Kappa:", max(kappa_values), "\n")

#-------------------------------------
# Set up cross-validation with internal resampling
#-------------------------------------
# Use sampling = "up" inside trainControl so that each fold is balanced separately
ctrl <- trainControl(
  method = "cv", 
  number = 3,
  sampling = "up" 
)

# Train k-NN with automatic k selection
# We use a sample of the original train_data to ensure validation is on real distributions
train_small <- train_data[sample(nrow(train_data), 10000), ] 

knn_cv <- train(Diabetes_012 ~ .,
                data = train_small,
                method = "knn",
                trControl = ctrl,
                preProcess = c("center", "scale"),
                tuneGrid = data.frame(k = 1:15),
                metric = "Kappa")

# Results
print(knn_cv)
plot(knn_cv, main = "Cross-Validation Kappa vs k (Internal Upsampling)")

# Best model predictions
knn_cv_pred <- predict(knn_cv, newdata = test_data)
confusionMatrix(knn_cv_pred, test_data$Diabetes_012)

#==============================================================================
# SAVE WORKSPACE 
#==============================================================================
save(list = ls(all.names = TRUE), file = "workspace2.RData")