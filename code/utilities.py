from evolutionai import StorageEngine
import pandas as pd
import json
import re

def clean_up_txt(page_txt):
    page_txt = page_txt.lower()
    page_txt = re.sub('\s+',' ',page_txt)
    page_txt = re.sub('[^0-9a-zA-Z]+', " ", page_txt)
    return page_txt


def read_descriptions():
    df = pd.read_csv('../data/sic_descriptons.tsv', sep='\t', names = ["class_num", "class_txt", "json"])
    json_formatter = lambda x: json.loads(x)
    df['json'] = df['json'].apply(json_formatter)
    return df

def find_intersection_of_classes(des_df, df):
    # ["class_num", "class_txt", "json"]
    classes_desc = set(des_df["class_num"])
    classes_web = set(df["label_num"])
    intersection = classes_desc.intersection(classes_web)
    return intersection


def load_domain_data():
    df = pd.read_csv('../data/domains.tsv', sep='\t', names = ["company_name", "company_id", "url", "vertical"])
    df = df[df.vertical != "None Supplied"]
    label_num, label_txt = [], []
    for ver in df["vertical"]:
        label_n, label_t = ver.split("-",1)
        label_num.append(int(label_n))
        label_txt.append(label_t)
    df["label_num"] = label_num
    df["label_txt"] = label_txt
    des_df = read_descriptions()
    intersection =find_intersection_of_classes(des_df, df)
    des_df = des_df[des_df["class_num"].isin(intersection)]
    df = df[df["label_num"].isin(intersection)]
    return des_df, df


#########################
#
# Find the classes that have detail or inclusion
#
########################
def des_txt_and_class_filtering(des_df, df):
    des_data = []
    des_labels = []
    used_classes = set()
    for des_json, cl_num in zip(des_df['json'], des_df["class_num"]):
        valid_txt = ""
        if ("detail" in des_json.keys()) or ("includes" in des_json.keys()) :
            used_classes.add(cl_num)
            for key in des_json:
                if key!="excludes":
                    if type(des_json[key])==list:
                        text_buffer = " "
                        for bullet in des_json[key]:
                            text_buffer += " " + bullet
                    else:
                        text_buffer = des_json[key]
                    valid_txt += " "+text_buffer
            valid_txt = clean_up_txt(valid_txt)
            des_data.append(valid_txt)
            des_labels.append(cl_num)
    des_df = des_df[des_df["class_num"].isin(used_classes)]
    des_df["txt"] = des_data
    des_df["new_l"] = des_labels
    df = df[df["label_num"].isin(used_classes)]
    return des_df, df

def query_web_data(df, size=None):
    storage = StorageEngine("/nvme/webcache_old/")
    web_sites = []
    labels = []
    summaries = []
    company_id = []
    all_urls = []
    print("Fetch websites from database")
    counter = 0
    for i, l, c_id in zip(df['url'], df["label_num"], df["company_id"]):
        if size !=None:
            counter +=1
            if counter > size:
                break
        # query database and get page object
        page = storage.get_page(i)
        # some domains are not scrapped
        try:
            page_txt = page.textSummary

            summaries.append(re.sub('\s+',' ',page_txt))
            # some preprocessing
            page_txt = clean_up_txt(page_txt)
            # page_txt = remove_url_chars(page_txt)
            web_sites.append(page_txt)
            company_id.append(c_id)
            labels.append(l)
            all_urls.append(i)
        except:
            pass
    df_web = pd.DataFrame()
    df_web["class_num"] = labels
    df_web["class_txt"] = web_sites
    df_web["summaries"] = summaries
    df_web["company_id"]= company_id
    df_web["urls"]=all_urls
    print("Labeled websites are {0}".format(len(df_web)))
    return df_web

def data_pipeline(size):
    des_df, df = load_domain_data()
    des_df, df = des_txt_and_class_filtering(des_df, df)
    df_web = query_web_data(df,size=size)
    print(len(df), len(df_web))

size=1000
data_pipeline(size)

