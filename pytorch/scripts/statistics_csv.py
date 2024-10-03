import pandas as pd
from scipy.stats import ttest_ind

input_file = 'pytorch-train-throughput-v2-fp16_merged'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file + '.csv')

# Define the columns of interest
columns_of_interest = ['resnet50', 'ssd', 'gnmt', 'bert_base_squad', 'bert_large_squad', 'tacotron2', 'waveglow', 'ncf']

# columns_of_interest = ['gnmt']


# Group the data by 'brand' and calculate mean and variance
# Group by 'brand' and print the values
grouped = df.groupby('brand')[columns_of_interest]

# for name, group in grouped:
#     print(f"\nBrand: {name}")
#     print(group['gnmt'])

grouped = df.groupby('brand')[columns_of_interest].agg(['mean', 'std']).round(3)

# print(grouped)

# Separate the data into two groups
dell_data = df[df['brand'] == 'DELL'][columns_of_interest]
smc_data = df[df['brand'] == 'SMC'][columns_of_interest]

# Perform a t-test for each benchmark task
ttest_results = {col: ttest_ind(dell_data[col], smc_data[col], equal_var=False) for col in columns_of_interest}

# Create a DataFrame for the t-test results
ttest_df = pd.DataFrame({
    't-statistic': {col: result.statistic for col, result in ttest_results.items()},
    'p-value': {col: result.pvalue for col, result in ttest_results.items()},
    'Significance': {col: 'Significant' if result.pvalue <= 0.05 else 'Not Significant' for col, result in ttest_results.items()}
})

# Save the grouped statistics to a CSV file
grouped.to_csv('grouped_statistics-' + input_file + '.csv')

# Save the t-test results to a CSV file
ttest_df.to_csv('ttest_results-' + input_file + '.csv', index_label='Benchmark Task')