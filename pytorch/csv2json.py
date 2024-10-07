import os
import pandas as pd
import glob
import json

DATA_PATH = './csv_v1_v2'
DATA_FORMAT = '.csv'
SKIPROWS = None

pd.set_option("display.max_colwidth", 1000)

for f in glob.glob(os.path.join(DATA_PATH, '*' + DATA_FORMAT)):
    f_out = os.path.splitext(f)[0] + '.json'
    
    # Read the CSV file
    df = pd.read_csv(f, index_col=False, skiprows=SKIPROWS, header=0)
    
    # Strip whitespace from all string entries in the dataframe
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    # Convert all columns (except the first) to numeric, forcing invalid parsing to NaN
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    # Save the dataframe as JSON, ensuring numbers are treated as numbers in JSON
    df.to_json(f_out, orient='records', double_precision=10)
