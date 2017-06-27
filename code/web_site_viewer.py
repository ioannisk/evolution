from evolutionai import StorageEngine
# from fetch_web_data import find_intersection_of_classes
from parse_descriptions import read_descriptions
import pandas as pd

def find_intersection_of_classes():
    # ["class_num", "class_txt", "json"]
    classes_desc = set(des_df["class_num"])
    classes_web = set(df["label_num"])
    intersection = classes_desc.intersection(classes_web)
    return intersection


storage = StorageEngine("/nvme/webcache/")
df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
# get path to the database
# read the domains.tsv file in pandas
print("Read domains")
df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
des_df = read_descriptions()
# remover unlabelled domains
df = df[df.vertical != "None Supplied"]
label_num, label_txt = [], []
# parse the label number and label txt seperatly
for ver in df["vertical"]:
    label_n, label_t = ver.split("-",1)
    label_num.append(int(label_n))
    label_txt.append(label_t)
df["label_num"] = label_num
df["label_txt"] = label_txt
# Keep only the descriptions that exist in the dataset
# intersection =find_intersection_of_classes()
# des_df = des_df[des_df["class_num"].isin(intersection)]
# df = df[df["label_num"].isin(intersection)]

web_sites = []
labels = []
summaries = []
company_id = []
print("Fetch websites from database")
counter = 0
for i, l, c_id in zip(df['url'], df["label_num"], df["company_id"]):
    counter +=1
    if counter > 10000:
        break
    # query database and get page object
    page = storage.get_page(i)
    # some domains are not scrapped
    try:
        page_txt = page.textSummary
        summaries.append(re.sub('\s+',' ',page_txt))
        page_txt = clean_up_txt(page_txt)
        web_sites.append(page_txt)
        company_id.append(c_id)
        labels.append(l)
    except:
        pass
print("Vectorize documents")
df_web = pd.DataFrame()
df_web["class_num"] = labels
df_web["class_txt"] = web_sites
df_web["summaries"] = summaries
df_web["company_id"]= company_id
print("Labeled websites are {0}".format(len(df_web)))
