# IBM HR Analytics Employee Attrition Analysis
This repository contains Python code for analyzing employee attrition using the IBM HR Analytics dataset. The dataset provides information about employees, their work environment, and whether they left the company (attrition) or stayed. The analysis involves data preprocessing, exploration, feature engineering, and the training of machine learning models for predicting employee attrition.

## Table of Contents
1. [Introduction](Introduction)
2. [Dataset](Dataset)
3. [Exploratory Data Analysis](Exploratory-Data-Analysis)
4. [Feature Engineering](Feature-Engineering)
5. [Data Visualization](Data-Visualization)
6. [Model Training](Model-Training)
7. [Results](Results)
8. [Conclusion](Conclusion)
9. [Usage](Usage)

## Introduction
Employee attrition, or the departure of employees from an organization, is a critical concern for businesses. Understanding the factors contributing to attrition can help in the development of strategies to retain valuable employees. This project explores the IBM HR Analytics dataset, aiming to gain insights into the factors influencing employee attrition and build machine learning models for prediction.

## Dataset
The dataset used in this analysis is the [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data) dataset obtained from Kaggle. It contains information about 1470 employees and 35 features, including age, daily rate, job satisfaction, and more.

## Exploratory Data Analysis
The initial exploration of the dataset includes importing required libraries, fetching data using Kaggle API key, and examining the dataset's structure. Descriptive statistics, such as the number of unique values and data types for each column, are also presented.

## Feature Engineering
The feature engineering section addresses missing values and duplicated entries. In this case, as there are no missing values or duplicated entries, no operations were performed. The dataset is then preprocessed by dropping columns with low counts of unique values, and categorical columns are encoded using one-hot encoding.

## Data Visualization
Data visualization is crucial for understanding the distribution of various features and their impact on attrition. Visualizations include count plots and a correlation matrix to identify relationships between features.

## Model Training
Machine learning models are trained using the preprocessed dataset. The imbalance in the dataset is addressed using Synthetic Minority Over-sampling Technique (SMOTE) to generate synthetic data. Models trained include Random Forest, XGBoost, Support Vector Machine (SVM), CatBoost, and Logistic Regression.

## Results
The results of each model, including accuracy, precision, recall, and F1 score, are presented. Evaluation plots, such as confusion matrices, ROC curves, and precision-recall curves, are also provided.
