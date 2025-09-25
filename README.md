# Data Cleaning & Preprocessing

## Objective
To clean and preprocess the Titanic dataset to make it ready for machine learning.

## Steps Performed
- Removed missing values from 'Age' and 'Embarked'
- Dropped the 'Cabin' column
- Converted categorical features to numeric using One-Hot Encoding
- Standardized 'Age' and 'Fare' using StandardScaler
- Visualized and removed outliers using boxplots and IQR method

## Tools Used
- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## Dataset Source
- Titanic Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset
- Original Dataset: Titanic-Dataset.csv
- Cleaned Dataset: Titanic-Dataset-cleaned.csv

## Fixes Applied
- Fixed chained assignment warning in Pandas by using direct assignment
