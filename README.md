# 📊 Credit Risk Classification

## 🔍 Overview

This project leverages supervised machine learning techniques to predict the credit risk of borrowers using historical lending data from a peer-to-peer lending platform. By building a logistic regression model, the goal is to classify loans as either:

- `0` – Healthy (low risk)
- `1` – High risk (likely to default)

This kind of analysis is critical for lenders who need to make informed decisions about offering credit.

---

## 🧠 Machine Learning Process

The following steps were performed as part of the modeling workflow:

1. Imported and cleaned the data from `lending_data.csv`
2. Separated the features (`X`) from the target labels (`y`)
3. Split the dataset into training and testing sets using `train_test_split`
4. Trained a **Logistic Regression** model using `scikit-learn`
5. Evaluated the model with a confusion matrix and classification report

---

## 📈 Model Performance

### Logistic Regression Results

- **Accuracy**: ~94%
- **Precision (Class 0 – Healthy Loans)**: ~0.94  
- **Recall (Class 0 – Healthy Loans)**: ~1.00  
- **Precision (Class 1 – High-Risk Loans)**: ~0.85  
- **Recall (Class 1 – High-Risk Loans)**: ~0.02  

> ⚠️ Note: The dataset is highly imbalanced, which significantly impacts the model's ability to predict high-risk loans.

---

## 🧾 Interpretation

While the logistic regression model demonstrates excellent performance in identifying healthy loans, it performs poorly in detecting high-risk loans — which is arguably the more important objective in credit risk assessment. The model rarely predicts class `1`, leading to a low recall for high-risk borrowers.

---

## ✅ Recommendation

At this stage, the model **should not be used in production** without further tuning and preprocessing. To improve performance:

- Address class imbalance using:
  - **SMOTE** (Synthetic Minority Over-sampling)
  - **Class weights** or **undersampling**
- Explore more robust models such as:
  - **Random Forest**
  - **Gradient Boosting (e.g., XGBoost)**
- Apply feature engineering and normalization

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## 📬 Contact

For any questions or feedback, feel free to reach out via GitHub.

---



