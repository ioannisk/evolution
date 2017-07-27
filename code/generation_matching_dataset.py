import pandas
import json
import random
import time
import os

MAX_LEN=111
MAX_DES_LEN=MAX_LEN
MAX_WEB_LEN=MAX_LEN


def web_des_intersection(class_descriptions, cmp_des):
    """Because of lenght restrictions
    the class_num set of web and des might not overlap.
    This function ensures that the training data will have a
    perfect overlap
    """
    print(len(cmp_des))
    des_set = set(class_descriptions.keys())
    web_set = set([cmp_des[key]["class_num"] for key in cmp_des])
    intersection =  des_set.intersection(web_set)
    class_descriptions = {key:class_descriptions[key] for key in class_descriptions if key in intersection}
    cmp_des = {key:cmp_des[key] for key in cmp_des if cmp_des[key]["class_num"] in intersection}
    des_set = set(class_descriptions.keys())
    web_set = set([cmp_des[key]["class_num"] for key in cmp_des])

    print(len(des_set))
    print(len(web_set))
    print(len(cmp_des))

    return class_descriptions, cmp_des

def read_descriptions():
    """Read the meta data file
    input: des txt file produced from wrt_locally.py
    output: {class_num: txt }
    """
    class_descriptions = {}
    with open("../data/descriptions_data_1.txt","r") as file_:
        for line in file_:
            line = line.strip()
            class_num , txt = line.split('\t')
            if len(txt.split()) <=MAX_DES_LEN:
                class_descriptions[class_num] = txt
    return class_descriptions

def read_meta():
    """Read the meta data file
    input: meta txt file produced from wrt_locally.py
    output: {id: {"class_num":, "txt": } }
    """
    companies_descriptions = {}
    with open("../data/web_site_meta_1.txt", "r") as file_:
        for line in file_:
            line = line.strip()
            id_, class_num , txt = line.split('\t')
            if len(txt.split()) <= MAX_WEB_LEN:
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


