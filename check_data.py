import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Symptom2Disease.csv")

# Show first 5 rows
print(df.head())

# Show column names
print(df.columns)

# Show number of rows and columns
print(df.shape)

# Check missing values
print(df.isnull().sum())