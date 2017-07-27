import pandas
import json
import random
import time
import os
from collections import Counter

MAX_LEN=111
MAX_DES_LEN=MAX_LEN
MAX_WEB_LEN=MAX_LEN

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

def web_des_intersection(class_descriptions, cmp_des):
    """Because of lenght restrictions
    the class_num set of web and des might not overlap.
    This function ensures that the training data will have a
    perfect overlap
    """
    des_set = set(class_descriptions.keys())
    web_set = set([cmp_des[key]["class_num"] for key in cmp_des])
    intersection =  des_set.intersection(web_set)
    class_descriptions = {key:class_descriptions[key] for key in class_descriptions if key in intersection}
    cmp_des = {key:cmp_des[key] for key in cmp_des if cmp_des[key]["class_num"] in intersection}
    return class_descriptions, cmp_des

def make_N_folds_classes(class_descriptions, companies_descriptions, N=5):
    """ Make N datasets such that there is no
    class overlap between training and testing.
    With some additional logic for dataset balance,
    so we can make sure that the splits have ~= #points
    """
    class_counts = Counter()
    print(len(companies_descriptions))
    for id_ in companies_descriptions:
        class_counts[companies_descriptions[id_]["class_num"]]+=1
    ranked = class_counts.most_common()
    ## [] instead of 0
    folds = [0 for i in range(N)]
    app_fold_volume = len(companies_descriptions)/N

    fold_index = 0
    for class_num, counts in ranked:
        if folds[fold_index] < app_fold_volume:
            folds[fold_index] += counts
        fold_index +=1
        if fold_index ==N:
            fold_index = 0
    print(folds)







    stop

    folds = []
    folds_counts = Counter()
    classes = list(class_descriptions.keys())
    split = int(len(classes)/N)
    folds = [classes[i*split:i*split+split] for i in range(N)]
    for id_ in companies_descriptions:
        for i, fold in enumerate(folds):
            if companies_descriptions[id_]["class_num"] in fold:
                folds_counts[i] +=1
    print(folds_counts)

def write_fold():
    path = "../data/folds"


if __name__=="__main__":
    class_descriptions = read_descriptions()
    companies_descriptions = read_meta()
    class_descriptions, companies_descriptions = web_des_intersection(class_descriptions, companies_descriptions)
    make_N_folds_classes(class_descriptions, companies_descriptions, N=5)



