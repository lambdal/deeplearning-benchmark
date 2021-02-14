import os

import pandas
import glob
import json

DATA_PATH = ''
DATA_FORMAT = '.csv'
SKIPROWS = None

pandas.set_option("display.max_colwidth", 1000)
for f in glob.glob(os.path.join(DATA_PATH, '*' + DATA_FORMAT)):
    f_out = os.path.splitext(f)[0] + '.json'
    df = pandas.read_csv(f,
                         index_col=False,
                         skiprows=SKIPROWS,
                         header=0)
    df.to_json(f_out, orient='records')
