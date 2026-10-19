# IBM-Assignment-03
# Employee Attrition Prediction Models

## Project Overview

This project develops machine learning classification models to predict employee attrition using the IBM HR Employee Attrition dataset. The goal is to identify patterns associated with employees leaving an organization and evaluate how machine learning can support HR decision-making.

## Objectives

- Explore and preprocess employee HR data.
- Build classification models to predict employee attrition.
- Evaluate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Improve model performance using hyperparameter tuning and cross-validation.
- Examine feature importance and interpret the results from a business perspective.

## Dataset

The project uses the IBM HR Employee Attrition dataset, containing 1,470 employee records and 35 original columns.

The target variable is `Attrition`:
- `Yes` = 1
- `No` = 0

The dataset includes employee demographics, job characteristics, compensation, satisfaction, and work-related information.

## Models Implemented

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

## Model Evaluation

Initial test-set results:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8605 | 0.6154 | 0.3404 | 0.4384 | 0.8115 |
| Decision Tree | 0.7551 | 0.2642 | 0.2979 | 0.2800 | 0.5700 |
| Random Forest | 0.8469 | 0.6250 | 0.1064 | 0.1818 | 0.7832 |
| XGBoost | 0.8673 | 0.7857 | 0.2340 | 0.3607 | 0.7973 |

After hyperparameter tuning, Logistic Regression achieved:

- **Accuracy:** 0.8605
- **Precision:** 0.6154
- **Recall:** 0.3404
- **F1-score:** 0.4384
- **ROC-AUC:** 0.8123

Tuning produced a small improvement in ROC-AUC. Logistic Regression achieved the best recall, F1-score, and ROC-AUC among the initial models, while XGBoost achieved the highest initial accuracy and precision.

## Business Interpretation

The results show that employee attrition prediction is challenging because the model must distinguish employees who leave from the larger group who stay.

The tuned Logistic Regression model achieved 34.04% recall for employees who left. This means it identified approximately one-third of actual attrition cases in the test set. Its ROC-AUC of 0.8123 indicates useful ability to distinguish between the two classes across classification thresholds.

The model could help HR teams explore attrition risk patterns and prioritize further investigation. Predictions should be used as decision-support information rather than as the sole basis for employee-related decisions.

## Limitations

- The dataset may not represent every organization or workforce.
- Attrition is imbalanced, so accuracy alone does not fully describe performance.
- Model associations do not establish the causes of employee attrition.
- Predictions should be reviewed for fairness, privacy, and potential bias.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Google Colab
- Matplotlib / Seaborn (if used for visualizations)

## Project Files

- `employee_attrition_prediction.ipynb` — notebook containing data preprocessing, model training, evaluation, and tuning.
- `business_report.md` — business interpretation and conclusions.
- `README.md` — project overview and results.

## Conclusion

Four machine learning classification models were developed and evaluated for employee attrition prediction. Logistic Regression provided the strongest overall balance of recall, F1-score, and ROC-AUC among the initial models, while XGBoost achieved the highest accuracy. Hyperparameter tuning provided a small ROC-AUC improvement. Further work could explore decision thresholds, class imbalance handling, and additional validation.

