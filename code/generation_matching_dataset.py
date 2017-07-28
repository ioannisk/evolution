import pandas
import json
import random
import time
import os
from collections import Counter

N = 10
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

#
# Not tested and most probably not working atm
#
# def make_N_folds_classes_equal_classes(class_descriptions, companies_descriptions):
#     """ Make N datasets such that there is no
#     class overlap between training and testing.
#     We need to make sure that the splits have ~= #classes
#     """
#     folds = []
#     folds_counts = Counter()
#     classes = list(class_descriptions.keys())
#     split = int(len(classes)/N)
#     folds = [classes[i*split:i*split+split] for i in range(N)]
#     for id_ in companies_descriptions:
#         for i, fold in enumerate(folds):
#             if companies_descriptions[id_]["class_num"] in fold:
#                 folds_counts[i] +=1
#     print(folds_counts)


def update_index(fold_index, folds, folds_volume):
    """ index needs to updated
    but if index >= N it needs to
    go back to 0
    """
    fold_index +=1
    if fold_index==N:
        fold_index = 0
        # folds = folds[::-1]
        # folds_volume = folds_volume[::-1]
    return fold_index, folds, folds_volume

def allocate_bin(folds, folds_volume, class_num, counts, fold_index, app_fold_volume):
    """ Recursive class bin allocator
    returns: N bin with the same amount of data points
    """
    # print("index: {}".format(fold_index))
    if folds_volume[fold_index] < app_fold_volume:
        folds[fold_index].append(class_num)
        folds_volume[fold_index] += counts
    else:
        fold_index, folds, folds_volume = update_index(fold_index, folds, folds_volume)
        allocate_bin(folds, folds_volume, class_num, counts, fold_index, app_fold_volume)
    fold_index, folds, folds_volume = update_index(fold_index, folds, folds_volume)
    return folds, folds_volume, fold_index

def make_N_folds_classes_equal_datapoints(class_descriptions, companies_descriptions):
    """ Make N datasets such that there is no
    class overlap between training and testing.
    We need to make sure that the splits have ~= #points
    """
    # Count and order classes according to datapoints
    class_counts = Counter()
    for id_ in companies_descriptions:
        class_counts[companies_descriptions[id_]["class_num"]]+=1
    # Rank according to least common count
    ranked = class_counts.most_common()[::-1]
    folds_volume = [0 for i in range(N)]
    folds = [[] for i in range(N)]
    # Maybe allow +5% in volyme if probelms
    app_fold_volume = len(companies_descriptions)/N
    fold_index = 0
    for class_num, counts in ranked:
        folds, folds_volume, fold_index = allocate_bin(folds, folds_volume, class_num, counts, fold_index, app_fold_volume)
    # make sure all training data is used
    assert (len(companies_descriptions))==(sum(folds_volume))
    # make sure all classes are used
    assert sum([len(i) for i in folds]) == len(class_descriptions.keys())
    print("Folds have volume of {}".format(folds_volume))
    print("Folds have #classes of {}".format([len(i) for i in folds]))
    return folds

def make_training_pairs():
    """ This function makes binary pairs
    so the decomposable attention can be trainined
    and evaluated in 2 classes (match, doesnt match)
    """





# def make_evaluation_pairs:
    """ This function makes as many pairs for a
    company as classes. This data is used for the final
    evaluation against all classes
    """


def write_fold():
    path = "../data/folds"


if __name__=="__main__":

    class_descriptions = read_descriptions()
    companies_descriptions = read_meta()
    class_descriptions, companies_descriptions = web_des_intersection(class_descriptions, companies_descriptions)
    folds = make_N_folds_classes_equal_datapoints(class_descriptions, companies_descriptions)



