# 💳 Credit Risk Assessment

Predicting the likelihood of loan default from borrower and loan characteristics, using Python and machine learning.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)


---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Dataset](#-dataset)
4. [Repository Structure](#-repository-structure)
5. [Methodology](#-methodology)
6. [Results](#-results)
7. [Key Insights](#-key-insights)
8. [Getting Started](#-getting-started)
9. [Limitations](#-limitations)
10. [Future Work](#-future-work)
11. [Author](#-author)
12. [License](#-license)

---

## 🔍 1. Project Overview

Lenders lose money when borrowers default. This project builds a classification model that estimates whether a loan applicant is likely to **default (1)** or **repay (0)**, based on applicant demographics, employment, credit history and loan attributes.

The full workflow lives in a single Jupyter notebook: data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## 🏦 2. Business Problem

A bank or fintech must decide **whom to lend to, how much, and at what interest rate**. Two errors matter:

| Error | Meaning | Business cost |
|---|---|---|
| False Negative | Risky borrower approved | Direct credit loss |
| False Positive | Good borrower rejected | Lost interest income and customer |

Because defaults are typically the minority class and a missed default is usually costlier than a lost good customer, the project emphasises **recall, precision, ROC-AUC and PR-AUC** rather than accuracy alone.

## 📊 3. Dataset

- **Dataset Link:**  [![Kaggle](https://img.shields.io/badge/Kaggle-%23035a7d.svg?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Target:** `loan_status` (0 = non-default, 1 = default)


| Column | Description |
|---|---|
| `person_age` | Applicant age (years) |
| `person_income` | Annual income |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | Employment length (years) |
| `loan_intent` | Purpose of loan (education, medical, venture, etc.) |
| `loan_grade` | Lender-assigned grade (A to G) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_status` | **Target**: 1 = default, 0 = repaid |
| `loan_percent_income` | Loan amount as a share of income |
| `cb_person_default_on_file` | Previous default on record (Y/N) |
| `cb_person_cred_hist_length` | Credit history length (years) |

## 📁 4. Repository Structure

```
Credit_Risk_Assessment/
├── Credit_Risk_Assesment.ipynb   # End-to-end analysis and modelling notebook
├── credit_risk_dataset.csv       # Raw dataset
├── .gitignore
├── LICENSE                       # MIT
└── README.md
```

## 🧪 5. Methodology

> Update this section so it matches exactly what your notebook does.

1. **Data cleaning**: inspect dtypes and missing values (commonly `person_emp_length`, `loan_int_rate`), remove duplicates, and treat impossible outliers (e.g., unrealistic ages or employment lengths).
2. **Exploratory data analysis**: target balance, distributions, default rate by grade, home ownership and loan intent, and a correlation heatmap.
3. **Preprocessing**: impute missing values, encode categorical features (one-hot / ordinal for `loan_grade`), scale numeric features where needed.
4. **Train/test split**: stratified on the target to preserve the default ratio.
5. **Modelling**: baseline Logistic Regression compared with tree-based models (Random Forest, Gradient Boosting / XGBoost).
6. **Evaluation**: confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC.

## 📈 6. Results

<!-- Replace with your real numbers from the notebook -->

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.92 | 0.82 | 0.80 | 0.81 | 0.87 |
| Random Forest | – | – | – | – | – |
| Gradient Boosting | – | – | – | – | – |

- Hyperparameter tuning with cross-validation
- Probability calibration and cost-based threshold selection
- Model explainability with SHAP

<img width="400" height="470" alt="SHAP summary" src="https://github.com/user-attachments/assets/a0da0d48-0e66-4c86-bfee-0cfa7e845e51" />

<img width="450" height="470" alt="SHAP waterfall" src="https://github.com/user-attachments/assets/8a83a109-ef22-452c-ace2-ed74645e7062" />




## 💡 7. Key Insights

<!-- Fill in from your analysis. Examples of the format: -->
- Default rate rises sharply from loan grade A to G.
- Higher `loan_percent_income` is associated with higher default risk.
- Borrowers who rent default more often than those who own or hold a mortgage.

## 🚀 8. Getting Started

**Prerequisites:** Python 3.9+

```bash
# 1. Clone
git clone https://github.com/rudrascience/Credit_Risk_Assessment.git
cd Credit_Risk_Assessment

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
# (add xgboost / imbalanced-learn if used in the notebook)

# 4. Launch
jupyter notebook Credit_Risk_Assesment.ipynb
```

Run all cells top to bottom. The notebook expects `credit_risk_dataset.csv` in the same folder.

## ⚠️ 9. Limitations

- Single static dataset; no out-of-time validation.
- Features such as age may raise fairness and regulatory concerns in real lending.
- Accuracy-style metrics can mislead on imbalanced data.
- Not production-ready: no deployment, monitoring or calibrated probabilities yet.

## 🔭 10. Future Work

- Fairness audit across demographic groups
- Scorecard (WoE/IV) model for interpretability
- Streamlit app for live scoring

## 👤 11. Author

**Rudrajit Das**
- GitHub: [@rudrascience](https://github.com/rudrascience)
- LinkedIn: [rudrajit-das](https://www.linkedin.com/in/rudrajit-das-32533b204)

## 📄 12. License

Released under the [MIT License](LICENSE).
