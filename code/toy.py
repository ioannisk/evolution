import pandas as pd

df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
df = df[df.vertical != "None Supplied"]
label_num, label_txt = [], []
for ver in df["vertical"]:
    label_n, label_t = ver.split("-",1)
    label_num.append(label_n)
    label_txt.append(label_t)
df["label_num"] = label_num
df["label_txt"] = label_txt
df['url'] = ["".join(["http://www.", u]) if u[:4] != "www." else "".join(["http://", u]) for u in df['url']]
for i in range(10):
    print(df['url'][i])

