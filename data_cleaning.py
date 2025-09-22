import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('Titanic-Dataset.csv')
print("First 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum()) # Display count of missing values per column

# Handle missing values
df.drop('Cabin', axis=1, inplace=True) #Drop 'Cabin' column due to high missing values
df['Age'] = df['Age'].fillna(df['Age'].median()) #Fill missing 'Age' with median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) #Fill missing 'Embarked' with mode

# Convert categorical to numeric
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True) # One-hot encode dropping first to avoid dummy variable trap

# Normalize Age and Fare
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']]) 

# Visualize outliers
sns.boxplot(data=df[['Age', 'Fare']]) 
plt.title('Boxplot for Age and Fare')
plt.show()

# Remove outliers using IQR method for 'Age' and 'Fare' columns
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25) # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75) # Third quartile (75th percentile)
    IQR = Q3 - Q1 # Interquartile range
    lower = Q1 - 1.5 * IQR # Lower bound
    upper = Q3 + 1.5 * IQR # Upper bound
    return df[(df[column] >= lower) & (df[column] <= upper)] # Filter out outliers

df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Fare')

# Save cleaned dataset
df.to_csv('Titanic-Dataset-cleaned.csv', index=False)
print("\n Data cleaned and saved to 'Titanic-Dataset-cleaned.csv'")
