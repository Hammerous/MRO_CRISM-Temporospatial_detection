import pandas as pd
import os

# Set the directory containing the CSV files
folder_path = r'Parameters_Summary'  # <-- Change this to your folder path
output_file = 'merged_Summary.csv'    # <-- Name of the output file

# List all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize a list to hold DataFrames
dataframes = []

# Loop through CSV files and read them
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Drop duplicate records
merged_df = merged_df.drop_duplicates()

# Save the result to a new CSV
merged_df.to_csv(os.path.join(folder_path, output_file), index=False)

print(f"Successfully merged {len(csv_files)} files into {output_file}.")
