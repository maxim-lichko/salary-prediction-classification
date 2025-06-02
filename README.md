# salary-prediction-classification

## Overview
This project aims to predict whether an individual earns more than $50,000 annually using a machine learning classification approach. The dataset, sourced from Kaggle, includes demographic, occupational, and socio-economic features. Emphasis is placed on model performance and interpretability using Explainable AI (XAI) techniques to ensure transparency, especially given the sensitive nature of features like race, gender, and education.

**Authors**: Maxim Lichko, Alexey Demchuk

## Dataset
The dataset contains 32,561 entries with 15 features, including:
- **Numerical Features**: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- **Categorical Features**: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- **Target Variable**: `salary` (`<=50K` or `>50K`)

The dataset is imbalanced, with approximately 76% of samples labeled `<=50K` and 24% labeled `>50K`.

## Methodology
1. **Exploratory Data Analysis (EDA)**:
   - Visualized numerical feature distributions using histograms and boxplots.
   - Analyzed categorical feature distributions with normalized bar plots.
   - Computed correlations using a heatmap for numerical features and Phi coefficients for binary features.

2. **Feature Engineering**:
   - Removed `education-num` due to redundancy with `education`.
   - Stripped whitespace from categorical columns.
   - Encoded `sex` and `salary` as binary variables.
   - Created `capital_diff` as the difference between `capital-gain` and `capital-loss`.
   - Grouped `native-country` into regions (e.g., Latin America, Asia) and `education` into broader categories (e.g., Bachelor_degree, Advanced_degree).
   - Unified `relationship` categories (e.g., Husband/Wife to Spouse).
   - Applied one-hot encoding to categorical features.
   - Dropped highly correlated features (e.g., `relationship_Spouse`, `race_White`) based on Phi coefficient analysis.

3. **Modeling**:
   - Split data into train (24,585 samples), validation (4,003 samples), and test (3,973 samples) sets with stratified sampling.
   - Evaluated three models: Decision Tree, Random Forest, and XGBoost.
   - Performed hyperparameter tuning using `RandomizedSearchCV` with ROC AUC as the scoring metric.
   - Optimized decision thresholds to maximize F1 score on the validation set.

4. **Model Evaluation**:
   - Metrics: ROC AUC, F1 score, precision, recall, and confusion matrix.
   - Compared models with and without relation-based features (e.g., `marital-status`, `relationship`).

5. **XAI Methods**:
   - Applied to the best-performing model (Random Forest without relation features).
   - **Permutation Importance**: Identified key features impacting ROC AUC.
   - **Partial Dependence Profiles (PDP)**: Visualized feature effects on predictions.
   - **SHAP**: Analyzed feature contributions to individual predictions using beeswarm, bar, and scatter plots.

## Results
### Models with Relation Features
| Model         | Test AUC ROC | Best F1 Score | Best Threshold | Accuracy | Precision (>50K) | Recall (>50K) |
|---------------|--------------|---------------|----------------|----------|------------------|---------------|
| Decision Tree | 0.9091       | 0.6723        | 0.608          | 0.84     | 0.64             | 0.79          |
| Random Forest | 0.9277       | 0.7026        | 0.638          | 0.87     | 0.73             | 0.73          |
| XGBoost       | 0.9329       | 0.7251        | 0.337          | 0.87     | 0.70             | 0.81          |

### Models without Relation Features
| Model         | Test AUC ROC | Best F1 Score | Best Threshold | Accuracy | Precision (>50K) | Recall (>50K) |
|---------------|--------------|---------------|----------------|----------|------------------|---------------|
| Random Forest | 0.8889       | 0.6419        | 0.583          | 0.84     | 0.66             | 0.69          |
| XGBoost       | 0.8967       | 0.6667        | 0.312          | 0.84     | 0.64             | 0.74          |

The Random Forest model without relation features was selected for XAI analysis due to its balance of performance and reduced reliance on sensitive features.

## Installation
1. Clone the repository:
   git clone https://github.com/maxim-lichko/salary-prediction-classification.git
2. Install dependencies:
   pip install -r requirements.txt
3. Download the dataset from Kaggle and place it in the data/ directory: https://www.kaggle.com/datasets/ayessa/salary-prediction-classification?resource=download

## Usage
Open the Jupyter notebook to preprocess data, train models, and generate XAI visualizations