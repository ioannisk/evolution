from evolutionai import StorageEngine
import pandas as pd
import re

def clean_up_txt(page_txt):
    page_txt  = re.sub('xa[0-9]+', " ", page_txt)
    # page_txt = re.sub('\s+',' ',page_txt)
    return page_txt




# get path to the database
storage = StorageEngine("/nvme/webcache/")
# read the domains.tsv file in pandas
df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
# remover unlabelled domains
df = df[df.vertical != "None Supplied"]
label_num, label_txt = [], []
# parse the label number and label txt seperatly
for ver in df["vertical"]:
    label_n, label_t = ver.split("-",1)
    label_num.append(label_n)
    label_txt.append(label_t)
df["label_num"] = label_num
df["label_txt"] = label_txt
# format the urls so the Storage Engine can be queried
# df['url'] = ["".join(["http://www.", u]) if u[:4] != "www." else "".join(["http://", u]) for u in df['url']]
# print(len(df['url']))
for i in range(10):
    # query database and get page object
    page = storage.get_page(df['url'][i])
    page_txt = page.textSummary
    page_txt = clean_up_txt(page_txt)
    print(page_txt)



# print(page.links)
