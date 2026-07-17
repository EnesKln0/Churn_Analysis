# Customer Attrition Analysis

This repository contains a machine learning pipeline designed to predict customer churn. The project explores the underlying dataset and evaluates the performance metrics of a custom-built Support Vector Classifier (SVC), a custom Random Forest, and an Artificial Neural Network (ANN).

## Exploratory Data Analysis (EDA)
Understanding the dataset structure, missing values, and feature correlations prior to modeling.

![Overall Churn Percentage](overall_churn_percentage.png)
![Percentage of Missing Values](percent_missing.png)
![Categorical Variables vs Churn Target](categorical_variables_vs_churn_target.png)
![Distribution of Numeric Variables by Churn](distribution_of_numeric_variables_by_churn.png)
![Correlation Matrix of Numeric Variables](correlation_matrix_of_numeric_variables.png)

---

## Final Model Evaluation
![Model Evaluation Dashboard](model_evaluation_dashboard.png)

---

## Custom Support Vector Classifier (SVC) Evaluation

### Quantitative Performance
* **Accuracy:** 0.7967
* **AUC Score:** 0.8175

### Performance Visualizations
![Custom SVC Optimal Probability Threshold](custom_svc_probability_cutoff.png)
![Custom SVC Confusion Matrix](custom_svc_confusion_matrix.png)
![SVM Calibration and Lambda](svm_calibration_and_lambda.png)

---

## Random Forest Evaluation

### Quantitative Performance
* **Accuracy:** 0.7925
* **AUC Score:** 0.8071

### Performance Visualizations
![Random Forest Optimization Dashboard](random_forest_optimization_dashboard.png)
![Random Forest Confusion Matrix](custom_random_forest_confusion_matrix.png)
![Random Forest Precision-Recall and n_trees](random_forest_precision_recall_and_n-trees.png)

---

## Artificial Neural Network (ANN) Evaluation

### Classification Report (Optimal Cutoff = 0.31)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Stay (0)** | 0.89 | 0.74 | 0.81 | 1033 |
| **Churn (1)** | 0.51 | 0.74 | 0.61 | 374 |
| **Overall Accuracy** | | | **0.74** | 1407 |

*Note: Hyperparameter tuning determined that 8 neurons in the hidden layer provided the best performance (AUC: 0.8263 over 3000 epochs).*

### Performance & Thresholds
![ANN Optimization Dashboard](ann_optimization_dashboard.png)
![Finding the Optimal Probability Cutoff](finding_the_optimal_probability_cutoff.png)
![ANN Final Matrix](ann_final_matrix.png)
![ANN Precision-Recall and Calibration](ann_precision_recall_and_calibration.png)

### Hyperparameter Tuning
![Effect of Neuron Count on ANN Performance](ann_neuron_count.png)
![Learning Rate Testing](learning_rate_testing.png)
![Train vs Validation Loss (Overfitting Check)](overfitting_check.png)
