# Credit Risk Classification

## Overview of the Analysis

The purpose of this analysis was to build a machine learning model that predicts the creditworthiness of loan applicants. The dataset, sourced from a peer-to-peer lending platform, contains historical lending data with borrower financial features and loan status outcomes.

The main objective was to predict the binary target variable `loan_status`, where:
- `0` indicates a healthy loan (low risk of default)
- `1` indicates a high-risk loan (likely to default)

We first examined the class balance using the `value_counts()` function, which revealed a significant class imbalance with far more healthy loans than high-risk loans. This imbalance poses a challenge for predictive modeling.

### Machine Learning Workflow:
- Imported and explored the dataset (`lending_data.csv`)
- Split the data into features (`X`) and target labels (`y`)
- Divided the data into training and testing sets using `train_test_split`
- Trained a logistic regression model using `LogisticRegression(random_state=1)`
- Evaluated the model's performance using:
  - Confusion matrix
  - Classification report (accuracy, precision, recall)

---

## Results

### Machine Learning Model 1: Logistic Regression

- **Accuracy**: ~94%  
- **Precision (Class 0 - Healthy Loans)**: ~0.94  
- **Recall (Class 0 - Healthy Loans)**: ~1.00  
- **Precision (Class 1 - High-Risk Loans)**: ~0.85  
- **Recall (Class 1 - High-Risk Loans)**: ~0.02  

*Note: Values may vary slightly depending on system and sklearn version.*

---

## Summary

The logistic regression model achieved high overall accuracy and was very effective in predicting healthy loans (`0`). However, its ability to detect high-risk loans (`1`) was extremely poor, with a recall of just 2%. This performance issue is largely due to the class imbalance in the dataset.

### Suggested Next Steps

- Address class imbalance using:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Random oversampling or undersampling
  - Using `class_weight='balanced'` in the logistic regression model
- Explore alternative models such as:
  - Random Forest
  - Gradient Boosting (e.g., XGBoost)
- Apply feature engineering to improve model performance


