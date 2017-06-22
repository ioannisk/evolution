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

count = 0
for txt, js in  zip(df["class_txt"], df["json"]):
    print(txt)
    for key in js:
        if key=="title":
            count +=1
        print(key + "::::",js[key])
    print("\n\n\n")
print(count)


# # df.join(df['json'].apply(json.loads).apply(pd.Series))
# for i in df['json']:
#     print(i['title'])
