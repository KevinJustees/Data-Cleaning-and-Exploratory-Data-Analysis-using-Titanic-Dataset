# Titanic Data Cleaning and EDA
# ----------------------------------------

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Show all columns in output
pd.set_option('display.max_columns', None)

# Step 2: Load the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Step 3: Basic info
print("\n--- Dataset Info ---")
print(train.info())

print("\n--- First 5 Rows ---")
print(train.head())

# Step 4: Check missing values
print("\n--- Missing Values ---")
print(train.isnull().sum())

# Step 5: Handle missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train.drop('Cabin', axis=1, inplace=True)

# Step 6: Verify missing values again
print("\n--- Missing Values After Cleaning ---")
print(train.isnull().sum())

# Step 7: Summary statistics
print("\n--- Summary Statistics ---")
print(train.describe())

# Step 8: Survival counts
print("\n--- Survival Counts ---")
print(train['Survived'].value_counts())

# Step 9: EDA Visualizations
sns.set(style="whitegrid")

# 1️⃣ Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Sex', hue='Survived', palette='Set2')
plt.title('Survival Count by Gender')
plt.show()

# 2️⃣ Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Pclass', hue='Survived', palette='Set3')
plt.title('Survival Count by Passenger Class')
plt.show()

# 3️⃣ Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(train['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.show()

# 4️⃣ Age vs. Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=train, palette='pastel')
plt.title('Age vs Survival')
plt.show()

# 5️⃣ Correlation Heatmap
plt.figure(figsize=(8,6))
corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Step 10: Key Insights
print("\n--- Key Insights ---")
print("1. Females had a higher survival rate than males.")
print("2. Passengers in 1st class were more likely to survive.")
print("3. Younger passengers tended to have slightly better survival chances.")
print("4. Fare and class are strongly related to survival.")
