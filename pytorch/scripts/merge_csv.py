import pandas as pd

# Read the two CSV files into DataFrames
df1 = pd.read_csv('mapping.csv', index_col='IP')
df2 = pd.read_csv('pytorch-train-throughput-v2-fp32.csv', index_col='IP')

# Merge the DataFrames on the 'IP' column
merged_df = pd.merge(df1, df2, on='IP', how='outer')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('pytorch-train-throughput-v2-fp32_merged.csv')
