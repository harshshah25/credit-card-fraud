# Credit Card Fraud Detection Project
## Introduction
This project tackles the challenge of identifying fraudulent credit card transactions by leveraging multiple machine learning algorithms. With the rise of online and cashless payments, accurately detecting fraud is crucial for maintaining transaction security. The dataset used in this project is notably imbalanced, simulating real-world scenarios where fraudulent cases are rare but high-impact. Through data balancing techniques and advanced models, this project aims to enhance fraud detection precision and reduce false alarms effectively.

## Project Goals
- Develop a model to distinguish between legitimate and fraudulent transactions.
- Implement Random Forest and XGBoost models to achieve high accuracy and minimize false positives.
- Address the class imbalance with SMOTE, enhancing model robustness and reliability.

## Dataset Information
The dataset, available here, contains anonymized credit card transactions by European cardholders. The dataset is taken from Kaggle - Credit Card Fraud Detecction.
Transactions: 284,807 in total, with only 492 marked as fraudulent, which constitutes 0.172% of the dataset.

## Features:
Time: Seconds elapsed between each transaction.
Amount: The monetary value of each transaction.
V1 to V28: Features resulting from PCA transformation for data privacy.
Class: Target variable (1 for fraud, 0 for genuine).
Due to the low percentage of fraudulent transactions, standard accuracy is not a sufficient metric. Instead, there is a focus on metrics like Precision-Recall AUC for evaluating model effectiveness.

# Key Components:
## Data Preprocessing
Class Imbalance Handling: SMOTE is applied to oversample minority class instances.
Feature Scaling: Standardization of features for optimized model performance.

## Machine Learning Models
Decision Tree: Basic classification algorithm providing interpretable results.
Random Forest: Ensemble method known for robust performance and feature importance.
XGBoost: Gradient boosting model, well-suited for imbalanced datasets.
Logistic Regression: Benchmark model for binary classification.
K-Nearest Neighbors (KNN): Non-parametric model that leverages similarity between samples.
Linear Discriminant Analysis (LDA): Statistical classifier that models the difference between classes.
Naive Bayes: Probabilistic model based on Bayes' theorem, suitable for small sample size.

## Evaluation Metrics
Confusion Matrix: Detailed insight into prediction performance.
Precision-Recall AUC: Suitable for imbalanced datasets to highlight model precision.
F1 Score and ROC AUC: Provide additional perspectives on model quality.

## Project Workflow
- Load and Preprocess Data: Import and clean data, handle class imbalance, and normalize features.
- Train Models: Train Random Forest and XGBoost models on the balanced dataset.
- Model Evaluation: Evaluate models using Precision-Recall AUC, F1 Score, and confusion matrices.
- Analyze Results: Compare model performance and identify the best model for fraud detection.

## Results
The models were evaluated based on Precision-Recall AUC due to the highly imbalanced dataset. By applying SMOTE, it significantly improved the performance of the models on minority class instances. This approach demonstrates the effectiveness of combining ensemble methods with sampling techniques for fraud detection.

## Key Insights
- Handling imbalanced data is essential in fraud detection to avoid misleading accuracy metrics.
- The combination of Random Forest and XGBoost, supported by SMOTE for class balancing, provides a strong framework for real-world fraud detection scenarios.
- This project highlights practical methods and performance evaluations suited to imbalanced datasets, paving the way for robust, scalable fraud detection solutions.
