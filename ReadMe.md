

# Telco Customer Churn Prediction

This project aims to predict which customers are likely to stop using a fictional telecom company’s services. By leveraging machine learning techniques, the objective is to develop a model that can help the business proactively retain customers and reduce churn rates.


## Project Goal

Build a classification model that predicts customer churn to:

* Understand patterns and factors contributing to churn
* Help the business reduce customer attrition



## Dataset Overview

* **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Size**: 7043 rows × 21 columns
* **Description**: Each row represents a customer with demographic information, account details, and service usage patterns.
* **Target Variable**: `Churn` – Indicates whether the customer has left the company in the last month.

### Key Features

* **Demographics**: Gender, SeniorCitizen, Partner, Dependents
* **Account Information**: Tenure, Contract type, Payment method, Paperless billing, Monthly and Total charges
* **Services**: Phone service, Internet, Online security, Backup, Streaming, and Technical support

## Data Preprocessing

### Initial Cleaning

* Converted `TotalCharges` from object to float after handling whitespace and missing values
* Dropped 11 rows with missing or zero `tenure` and `TotalCharges`
* Removed duplicates

### Encoding

* **Label Encoding**: Applied to binary and ordinal categorical features
* **One-Hot Encoding**: Used for nominal features like `PaymentMethod`, `Contract`, and `InternetService`

### Scaling

* Standardized numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`


### Balancing

* Addressed class imbalance (73% "No", 27% "Yes") using `RandomOverSampler`



## Modeling

### Algorithms Used

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* XGBoost
* Support Vector Machine (SVM)

### Evaluation Metric

Used **Accuracy**, **Precision**, **Recall**, and **F1-score**, with special attention to the *minority class (churned customers)*.

| Model             | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| ----------------- | -------- | ----------------- | -------------- | ---------------- |
| **Random Forest** | 77%      | 0.57              | 0.56           | 0.56             |
| XGBoost           | 74%      | 0.52              | 0.63           | 0.57             |
| Logistic Reg.     | 73%      | 0.50              | **0.79**       | 0.61             |
| SVM               | 73%      | 0.50              | 0.74           | 0.60             |
| Decision Tree     | 72%      | 0.47              | 0.49           | 0.48             |
| KNN               | 70%      | 0.46              | 0.67           | 0.54             |

### Observations

* **Random Forest** delivered the best overall performance.
* **Logistic Regression** achieved the **highest recall (79%)** on churners, indicating it’s more sensitive to detecting potential losses.
* Models struggled with class imbalance, but oversampling improved recall significantly.



## Key Insights

* **Churn is highest** among month-to-month customers and those using **Fiber Optic** internet.
* Features like **contract type**, **tenure**, and **monthly charges** were strong predictors.
* Handling **imbalanced classes** and **feature scaling** was crucial for meaningful model performance.


## Tech Stack

* **Python**, **Pandas**, **NumPy**, **Scikit-learn**, **XGBoost**, **Imbalanced-learn**, **Matplotlib**, **Seaborn**
* **Preprocessing**: Encoding, Scaling, Balancing
* **Modeling**: Classification algorithms, performance evaluation


## Next Steps

* Explore **ensemble models** or **stacking classifiers** for further performance gains
* Tune hyperparameters using **GridSearchCV**
* Apply **SHAP values** for model interpretability
* Build a **dashboard** to monitor churn KPIs in real time
