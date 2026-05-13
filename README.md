# Heart Disease Analysis & Prediction

## Overview
Heart disease is the leading cause of death in the United States. This project uses machine learning to predict whether an individual is at risk of a heart attack based on behavioral, demographic, and health data collected by the CDC.

The dataset contains over **445,000 records and 40 attributes** from a 2022 CDC survey across all 50 states, sourced from Kaggle. The target variable `HadHeartAttack` presented a significant class imbalance of 16.6:1, making this a challenging and realistic binary classification problem.

## Dataset
- **Source:** [Kaggle - Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/)
- **Records:** 445,000+
- **Features:** 40 (mix of qualitative and quantitative)
- **Target Variable:** `HadHeartAttack` (Binary: Yes / No)

## Tech Stack
- **Language:** Python
- **Environment:** Google Colab / Jupyter Notebook
- **Libraries:** scikit-learn, pandas, NumPy, matplotlib, seaborn

### Preprocessing
- Filled missing numerical values with the **median** to avoid outlier skew
- Filled missing categorical values with an **"Unknown" placeholder**
- Dropped features with **40%+ missing values**
- Applied **Label Encoding** to convert categorical features to numerical values
- Used **StandardScaler** to normalize numerical features
- Built a **correlation matrix** to check for multicollinearity
- Addressed class imbalance via **downsampling** to a 1:1 ratio

### Models Implemented
| Model | Accuracy | AUC-ROC |
|---|---|---|
| Decision Tree | ~82% | 0.873 |
| Random Forest | ~80% | 0.884 |
| Logistic Regression | ~79% | 0.885 |
| Support Vector Machine (LinearSVC) | ~80% | 0.882 |

- Used **Randomized Search Cross Validation** with **Stratified K-Fold** for hyperparameter tuning across all models
- Evaluated using **Precision, Recall, F1 Score, and AUC-ROC** to account for class imbalance

## Key Findings
- **HadAngina** was the strongest predictor of heart attack risk across all four models — chest discomfort caused by reduced blood flow to the heart is a direct precursor to heart attacks
- **AgeCategory, HadStroke, ChestScan,** and **Sex** were consistently among the top predictors
- Surprisingly, **BMI, smoking status, and alcohol consumption** ranked among the lower predictors
- All four models achieved AUC-ROC scores between **0.873 and 0.885**, demonstrating strong and consistent predictive signal

## Results Summary
All models significantly outperformed a naive majority-class classifier (which would achieve 94% accuracy but 0% heart attack recall), validating the importance of proper evaluation metrics under class imbalance.

## Future Directions
- Explore **SMOTE** for handling class imbalance
- Implement **XGBoost** for potential performance gains
- Investigate **feature reduction** — removing HadAngina to allow other features more predictive weight
- Expand to multi-class risk stratification

## Files
- `Heart_Disease_Analysis.ipynb` — Full analysis notebook with code, visualizations, and results
- `Heart_Disease_Analysis_Report.pdf` — Detailed 6-page project report
