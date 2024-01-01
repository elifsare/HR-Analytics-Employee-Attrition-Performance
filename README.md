# Employee Attrition Analysis
This repository contains Python code for analyzing employee attrition using the IBM HR Analytics dataset. The dataset provides information about employees, their work environment, and whether they left the company (attrition) or stayed. The analysis involves data preprocessing, exploration, feature engineering, and the training of machine learning models for predicting employee attrition.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering](#feature-engineering)
5. [Data Visualization](#data-visualization)
6. [Model Training](#model-training)
7. [Results](#results)

## Introduction
This repository presents a comprehensive analysis of employee attrition using the IBM HR Analytics dataset. Employee attrition, or the departure of employees from an organization, is a critical aspect that significantly impacts business performance. By understanding the underlying factors contributing to attrition, businesses can implement strategies to retain valuable employees. This project utilizes various data science techniques, including data preprocessing, exploratory data analysis (EDA), feature engineering, and the training of machine learning models to predict and understand employee attrition patterns.

## Dataset
The dataset used in this analysis is the [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data) dataset obtained from Kaggle. This extensive dataset contains information on 1470 employees, encompassing 35 features such as age, daily rate, job satisfaction, and more. The dataset serves as the foundation for uncovering insights into the dynamics of employee attrition within the context of different workplace factors.

## Exploratory Data Analysis
The initial phase involves importing necessary libraries and fetching the dataset using the Kaggle API key. The dataset's structure is examined through exploratory data analysis, providing an overview of its characteristics. Descriptive statistics, including the number of unique values and data types for each column, offer valuable insights into the dataset's composition.

## Feature Engineering
Feature engineering is a crucial step in preparing the dataset for model training. While addressing missing values and duplicated entries, it is essential to adapt the dataset to the specific needs of the analysis. In this case, the absence of missing values and duplicates streamlines the process. The dataset is further preprocessed by removing columns with low counts of unique values, and categorical columns undergo one-hot encoding. The code provided in the analysis uses the get_dummies function from the pandas library for one-hot encoding. Specifically, the following line of code is used to perform one-hot encoding on the categorical columns:
```python
df_encoded = pd.get_dummies(df[categorical_columns], columns=categorical_columns, drop_first=True)
```

## Data Visualization
Effective data visualization is integral to gaining a deeper understanding of feature distributions and their impact on attrition. Visualization techniques, such as count plots and a correlation matrix, illuminate relationships between different features. This visual exploration aids in identifying potential patterns and insights that contribute to the overall narrative of employee attrition. Visualizations include count plots and a correlation matrix to identify relationships between features.

## Model Training
Machine learning models play a central role in predicting and understanding employee attrition. Given the inherent imbalance in the dataset, Synthetic Minority Over-sampling Technique (SMOTE) is employed to generate synthetic data, ensuring a more balanced representation of both attrition and non-attrition cases. 
```python
smote = SMOTE(random_state=42, sampling_strategy = 0.5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```
Models, including Random Forest, XGBoost, Support Vector Machine (SVM), CatBoost, and Logistic Regression, are trained and evaluated to determine their effectiveness in predicting attrition.

## Results
The results section presents a comprehensive evaluation of each model's performance, covering metrics such as accuracy, precision, recall, and F1 score. Evaluation plots, including confusion matrices, ROC curves, and precision-recall curves, provide a visual representation of the models' predictive capabilities. This thorough analysis aims to equip stakeholders with actionable insights into employee attrition dynamics and facilitate informed decision-making for talent retention strategies.
