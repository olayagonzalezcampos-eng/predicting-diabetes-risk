# Statistical Learning: Implementation of Machine Learning Tools on the Diabetes Dataset

## Authors & Project Information
- **Authors:** Patricia Cabezón Díez, Olaya González Campos, Naroa Martín Casado
- **Context:** Statistical Learning Project — UCI Machine Learning Repository
- **Technologies:** R (Quarto, `caret`, `tidyverse`, `e1071`, `randomForest`, `xgboost`, `lightgbm`, `catboost`) & Python (`PyTorch`, `scikit-learn`, `pandas`, `seaborn`)

---

## Project Overview

This project implements and evaluates various Supervised Machine Learning methods to predict the categorical response variable `Diabetes_012` using demographic, lifestyle, and clinical risk factor indicators from the UCI Machine Learning Repository's **Diabetes Health Indicators** dataset.

The target variable is multiclass with three categories:
- **0 — No Diabetes** (~84% of original data)
- **1 — Prediabetes** (~2% of original data)
- **2 — Diabetes** (~14% of original data)

Due to severe class imbalance, standard accuracy is misleading (No Information Rate = 84.71%). The study focuses heavily on imbalance-robust metrics, prioritizing **Macro F1-Score**, **Cohen’s Kappa**, **Macro Balanced Accuracy**, and **Per-Class Recall**.

---

## Dataset Description & Preprocessing

### Dataset Specifications
- **Original Size:** 253,680 observations, 22 variables.
- **Representative Subsample:** 50,000 observations (`set.seed(1234)`) to balance computational performance and statistical power.
- **Missing Values:** 0 missing values across all features.
- **Feature Breakdown:**
  - **14 Binary Predictors:** `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex`.
  - **3 Ordinal Factors:** `Age`, `Education`, `Income`.
  - **1 Categorical Assessment:** `GenHlth`.
  - **3 Continuous Numerical Predictors:** `BMI`, `MentHlth`, `PhysHlth`.

### Data Preparation Pipeline
1. **Factor & Ordinal Encoding:** Conversion of categorical variables into factors with explicit ordering applied to `Age`, `Education`, and `Income`.
2. **Train/Test Split:** 80/20 random stratified split (40,000 training observations / 10,000 testing observations, `set.seed(123)`).
3. **Training Resampling (Upsampling):** Applied **exclusively to the training set** (`caret::upSample`) to achieve equal class representation. **The test set remains untouched (unbalanced)** to reflect realistic clinical diagnostic conditions.
4. **Feature Standardization:** Continuous numerical variables centered and scaled for distance-sensitive algorithms.

---

## Implemented Machine Learning Algorithms

The study evaluates four families:

### 1. Probabilistic & Discriminant Models
* **Multinomial Logistic Regression:** Evaluated both as an unadjusted baseline (84.86% overall accuracy, but 0.0% recall on Prediabetes due to majority-class bias) and with upsampling (63.30% accuracy, 33.3% Prediabetes recall, 64.6% Diabetes recall).
* **Linear Discriminant Analysis (LDA):** Achieved 61.47% accuracy and 52.00% macro sensitivity, with primary class separation driven by `Age`, `GenHlth`, `HighBP`, and `HighChol`.
* **Quadratic Discriminant Analysis (QDA):** Reached 59.11% accuracy with high Diabetes recall (79.7%), though Prediabetes recall dropped to 10.1% due to parameter estimation noise across class-specific covariance matrices.
* **Naive Bayes:** Achieved 66.99% overall accuracy under conditional independence assumptions.
* **Cost-Sensitive Decision Framework:** Applied expected risk refactoring using an asymmetric loss matrix $C$ ($C_{\text{Diabetes}, \text{No Diabetes}} = 3$), elevating Diabetes sensitivity to 91.0% to align model outputs with clinical diagnostic priorities.

### 2. Distance-Based Methods
* **k-Nearest Neighbors (k-NN):** Evaluated across continuous and standardized categorical feature representations, optimizing parameter $k$ via cross-validation to prevent minority class suppression along local distance boundaries.

### 3. Tree-Based & Ensemble Methods
* **Random Forest (`randomForest`):** `mtry` parameter tuned via Out-Of-Bag (OOB) error minimization, utilizing explicit per-class sample balancing (`sampsize`) across 200 trees.
* **XGBoost (`xgboost`):** Regularized gradient boosting optimized with a `multi:softmax` objective. Evaluated early stopping against an un-upsampled validation split (`X_val`) carved from original training data to avoid synthetic bias.
* **LightGBM (`lightgbm`):** Leaf-wise tree growth with histogram-based split finding for rapid and scalable multiclass learning.
* **CatBoost (`catboost`):** Ordered boosting using symmetric decision trees with native categorical variable optimization (`od_type = "Iter"`, `od_wait = 20`).

### 4. Neural Networks & Deep Learning
* **Multi-Layer Perceptron (PyTorch / scikit-learn):** Deep neural network architecture trained with cross-entropy loss, dropout regularization, and batch normalization to capture complex non-linear feature interactions.

---

## Repository Structure
```text
.
├── data/
│   └── dataset_diabetes.csv                 # Raw dataset from UCI repository
├── reports/
│   ├── Final_Assignment_Statistical_Learning_3.html # Second comprehensive HTML report
│   └── Project1_final.html                  # First comprehensive HTML report
├── scripts/
│   ├── Final_Assignment_Statistical_Learning.qmd    # Main Quarto source document
│   ├── Project1_final.qmd                   # Initial Quarto exploratory document
│   ├── knn_diabetes_dataset.R               # k-NN pipeline and evaluation script
│   └── trees_diabetes_dataset.R             # Tree-based algorithms evaluation script
├── workspaces and objects/
│   ├── class_labels.rds                     # Target class level definitions
│   ├── comparison_table.rds                 # Benchmark performance dataframe
│   ├── model_cat.rds                        # Serialized CatBoost model object
│   ├── model_lgb.rds                        # Serialized LightGBM model object
│   ├── model_rf.rds                         # Serialized Random Forest model object
│   ├── model_xgb.rds                        # Serialized XGBoost model object
│   ├── test_rf.rds                          # Processed test dataset for tree models
│   ├── workspace_final.RData                # Saved R environment session image
│   ├── workspace2.RData                     # Secondary R workspace image
│   └── X_test.rds                           # Processed test predictor matrix
├── .gitattributes                           # Git configuration settings
├── LICENSE                                  # Project licensing terms
└── README.md                                # Repository overview documentation
```
---

## Setup & Requirements

### 1. Required Software & R Environment

* **R** (version >= 4.0.0 recommended).
* **RStudio** or **Quarto CLI** to render `.qmd` documents into standalone HTML reports with floating table of contents (`toc`) and embedded self-contained resources (`embed-resources`).

### 2. Required Data & Pre-Trained Model Objects

To avoid re-running intensive grid searches during document compilation, make sure the root workspace contains:

* `dataset_diabetes.csv`
* `workspace_final.RData` and `workspace2.RData`
* Serialized model files: `model_rf.rds`, `model_xgb.rds`, `model_lgb.rds`, `model_cat.rds`, `comparison_table.rds`, `test_rf.rds`, `X_test.rds`, and `class_labels.rds`.

### 3. Package Dependencies (CRAN)

Execute the following R script to automatically install and load all required CRAN dependencies:

```r
packages <- c("tidyverse", "corrplot", "gridExtra", "scales", "MASS", "e1071", 
              "caret", "klaR", "GGally", "patchwork", "nnet", "effectsize", 
              "rcompanion", "naivebayes", "pROC", "class", "randomForest", 
              "xgboost", "lightgbm", "ggrepel", "rmarkdown", "knitr")

new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

```

### 4. Manual CatBoost Package Installation

The `catboost` package must be installed directly from the GitHub release binaries.

For **Windows (R 4.x)**:

```r
if (!require("remotes")) install.packages("remotes")
remotes::install_url(
  "[https://github.com/catboost/catboost/releases/download/v1.2.7/catboost-R-Windows-1.2.7.tgz](https://github.com/catboost/catboost/releases/download/v1.2.7/catboost-R-Windows-1.2.7.tgz)",
  INSTALL_opts = c("--no-multiarch", "--no-test-load")
)

```

*(Note for macOS/Linux users: Replace the URL above with the corresponding platform tarball from the [official CatBoost releases repository](https://github.com/catboost/catboost/releases)).*

