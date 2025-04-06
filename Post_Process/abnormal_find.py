import os
import glob
import pandas as pd

# Specify the folder containing the CSV files
folder_path = "Change_Detect"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# List to store anomalous records from each file
anomalies_list = []

for file in csv_files:
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Situation 1: 1 std deviation
        lower_1std = df['mean_val'] - df['std_val']
        upper_1std = df['mean_val'] + df['std_val']
        situation_1 = (df['first_quantile'] < lower_1std) | (df['first_quantile'] > upper_1std) | \
                    (df['third_quantile'] < lower_1std) | (df['third_quantile'] > upper_1std)

        # Situation 2: 3 std deviation
        lower_3std = df['mean_val'] - 3 * df['std_val']
        upper_3std = df['mean_val'] + 3 * df['std_val']
        situation_2 = (df['min_val'] < lower_3std) | (df['min_val'] > upper_3std) | \
                    (df['max_val'] < lower_3std) | (df['max_val'] > upper_3std)

        # Filter anomalies where either situation is True
        anomalies = df[situation_1 | situation_2].copy()

        # Add a column to indicate the situation
        anomalies['situation'] = ''
        anomalies.loc[situation_1, 'situation'] += '1'
        anomalies.loc[situation_2, 'situation'] += '2'

        # Drop rows where situation is still empty, if any (not likely here)
        anomalies = anomalies.dropna(subset=['situation'])

        # Add a column to track the source file
        anomalies['source_file'] = os.path.basename(file)

        # Append to the list if anomalies are found
        if not anomalies.empty:
            anomalies_list.append(anomalies)
    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Combine all anomalies into a single DataFrame and save to a new CSV file
if anomalies_list:
    result_df = pd.concat(anomalies_list, ignore_index=True)
    # output_file = os.path.join(folder_path, "anomalies.csv")
    output_file = "anomalies.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Anomalies saved to {output_file}")
else:
    print("No anomalies found across files.")
