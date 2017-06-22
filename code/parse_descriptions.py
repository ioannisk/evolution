import pandas as pd
import json

################################
#
# json keys
#
#
#
################################


df = pd.read_csv('../data/sic_descriptons.tsv', sep='\t', names = ["class_num", "class_txt", "json"])
json_formatter = lambda x: json.loads(x)
df['json'] = df['json'].apply(json_formatter)
keys  = set()
for i in df['json']:
    keys.update(set(i.keys()))
print(keys)



# # df.join(df['json'].apply(json.loads).apply(pd.Series))
# for i in df['json']:
#     print(i['title'])
