import pandas as pd

df = pd.read_csv('../data/sic_descriptons.tsv', sep='\t', names = ["class_num", "class_txt", "discriptions"])
print(df)

# df.join(df['json'].apply(json.loads).apply(pd.Series))
