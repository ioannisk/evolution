import pandas as pd
import json

df = pd.read_csv('../data/sic_descriptons.tsv', sep='\t', names = ["class_num", "class_txt", "json"])
for i in df['json']:
    print(i)

df.join(df['json'].apply(json.loads).apply(pd.Series))
for i in df['json']:
    print(i)
