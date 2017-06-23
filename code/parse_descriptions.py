import pandas as pd
import json

def read_descriptions():
    df = pd.read_csv('../data/sic_descriptons.tsv', sep='\t', names = ["class_num", "class_txt", "json"])
    json_formatter = lambda x: json.loads(x)
    df['json'] = df['json'].apply(json_formatter)
    return df

def check_json_keys():
    keys = set()
    df = read_description()
    for entry in df['json']:
        keys.update(set(entry.keys()))
    return (keys)

def make_class_dic():
    class_dic = {}
    df = read_description()
    for label in df['class_txt']:
        num, txt = label.split('-',1)
        class_dic[num] = txt
    return class_dic



print check_json_keys

# count = 0
# for txt, js in  zip(df["class_txt"], df["json"]):
#     print(txt)
#     checked = False
#     for key in js:
#         if key in ['includes', 'excludes', 'detail'] and not checked:
#             count +=1
#             checked = True
#         print(key + "::::",js[key])
#     print("\n\n\n")
# print(count)
# check_json_keys(df)

# # df.join(df['json'].apply(json.loads).apply(pd.Series))
# for i in df['json']:
#     print(i['title'])
