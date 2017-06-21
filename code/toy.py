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


print(df)

# df
# print(len(df1))

# [for company in companies_df["vertical"]]
# print(len(companies_df))
# for i in range(companies_df):
#     print(companies_df[i])
# companies_df['url'] = ["".join(["http://www.", u]) if u[:4] != "www." else "".join(["http://", u]) for u in companies_df['url']]

