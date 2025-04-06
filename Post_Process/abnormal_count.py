import pandas as pd

# 1. Open the CSV file
file_path = r'Post_Process\anomalies.csv'
df = pd.read_csv(file_path)

# 2. Count values in a specific column and sort from largest to smallest
field_to_count = 'band_name'  # Replace with your target column name
value_counts = df[df['situation'].isin([2, 12])][field_to_count].value_counts()
#value_counts = df[field_to_count].value_counts()

# 4e. Print the count result
print(len(value_counts))
print(value_counts)
