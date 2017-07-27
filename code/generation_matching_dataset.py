import pandas
import json
import random
import time
import os

MAX_LEN=111
MAX_DES_LEN=MAX_LEN
MAX_WEB_LEN=MAX_LEN

def web_des_intersection(class_descriptions, companies_descriptions):
    des_set = set(class_descriptions.keys())
    web_set = set([companies_descriptions[key]["class_num"] for key in companies_descriptions])
    print(len(des_set))
    print(len(web_set))


def read_descriptions():
    class_descriptions = {}
    with open("../data/descriptions_data_1.txt","r") as file_:
        for line in file_:
            line = line.strip()
            class_num , txt = line.split('\t')
            if len(txt.split()) <=MAX_DES_LEN:
                class_descriptions[class_num] = txt
    return class_descriptions


def read_meta():
    companies_descriptions = {}
    with open("../data/web_site_meta_1.txt", "r") as file_:
        for line in file_:
            line = line.strip()
            id_, class_num , txt = line.split('\t')
            companies_descriptions[id_] = {"class_num":class_num, "txt":txt}
    return companies_descriptions



if __name__=="__main__":
    class_descriptions = read_descriptions()
    companies_descriptions = read_meta()
    class_descriptions, companies_descriptions = web_des_intersection(class_descriptions, companies_descriptions)

    # b = set(b.keys())
    # print(len(b))
    # print(len(a))
    # print( b == a )


