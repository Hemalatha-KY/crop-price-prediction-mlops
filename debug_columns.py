import pandas as pd

# Load data
df = pd.read_csv('dataset_indian_crop_price.csv')

# Remove first column if it's unnamed
if df.columns[0] == 'Unnamed: 0':
    df = df.drop(df.columns[0], axis=1)

print("Target column:", repr(df.columns[-1]))
print("All columns:", df.columns.tolist())
