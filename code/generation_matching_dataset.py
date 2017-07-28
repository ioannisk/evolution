import pandas
import json
import random
import time
import os
from collections import Counter, defaultdict
import numpy as np

# Not actually 20 folds
# 2k in each bucket is convenient number for testing quickly
N = 20
MAX_LEN=111
MAX_DES_LEN=MAX_LEN
MAX_WEB_LEN=MAX_LEN

def write_json_line(json_ ,file_):
    json.dump(json_ , file_)
    file_.write('\n')


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
    data_path = "../data/folds/"
    with open(data_path+"volumes_data.txt","w") as file_:
        file_.write("Folds have volume of {}\n".format(zip(folds_volume, range(len(folds_volume)))))
        file_.write("Folds have #classes of {}\n".format(zip([len(i) for i in folds],range(len(folds_volume)))))
    return folds

def merge_folds(class_folds):
    """ Return N training, validation pairs from raw folds
    return: list of tuples (trainig, validation) for each fold
    """
    data = []
    class_folds = np.array(class_folds)
    for i, validation in enumerate(class_folds):
        mask = np.ones(len(class_folds), dtype=bool)
        mask[i] = 0
        training = (class_folds[mask])
        training = np.concatenate(training)
        data.append((training,validation))
    return data


def make_pairs(fold_classes,class_descriptions, companies_descriptions,classes_companies):
    """ Match the document with the
    des class if match or a random
    des class if not match. Then shuffle
    so MLP can learn.
    """
    positive = []
    negative = []
    ## positive pairs
    for class_ in fold_classes:
        companies = classes_companies[class_]
        class_des = class_descriptions[class_]
        for company in companies:
            company_des = companies_descriptions[company]["txt"]
            company_class = companies_descriptions[company]["class_num"]
            json_buffer = {'des':class_des, 'web':company_des, 'class':"entailment",
            'des_class':class_, 'web_class':company_class, 'web_id':company}
            positive.append(json_buffer)
    ## negative pairs
    # 2 design choices
    #       1. random sample a wrong company given a class
    #       2. random sample a wrong classs given a company (all companies are used)
    # for now implement 2
    for class_ in fold_classes:
        allowed_samples = list(fold_classes)
        allowed_samples.remove(class_)
        companies = classes_companies[class_]
        for company in companies:
            company_des = companies_descriptions[company]["txt"]
            company_class = companies_descriptions[company]["class_num"]
            sample_class = allowed_samples[random.randint(0, len(allowed_samples) - 1)]
            class_des = class_descriptions[sample_class]
            json_buffer = {'des':class_des, 'web':company_des, 'class':"contradiction",
            'des_class':sample_class, 'web_class':company_class, 'web_id':company}
            negative.append(json_buffer)
    ## shuffle data for learning purpose
    data = positive + negative
    random.shuffle(data)
    return data

#
# MAYBE!!!!!!! data leak in negations is a very smart idea
# It is not a mistake rather than an advantage of this classifier
# We know what some websites are not, not necessarily what they are
#
def make_training_dataset(class_folds, class_descriptions, companies_descriptions,classes_companies):
    """ This function makes binary pairs
    so the decomposable attention can be trainined
    and evaluated in 2 classes (match, doesnt match)
    and writes the data on disk
    """
    data_path = "../data/folds/"
    try:
        os.mkdir(data_path)
    except:
        pass
    for i, (training, validation) in enumerate(class_folds):
        print("Writting fold {}".format(i))
        training_pairs = make_pairs(training,class_descriptions, companies_descriptions,classes_companies)
        validation_pairs = make_pairs(validation,class_descriptions, companies_descriptions,classes_companies)
        path = data_path + "fold{}/".format(i)
        try:
            os.mkdir(path)
        except:
            pass
        with open(path+"training.json", "w") as file_:
            for pair in training_pairs:
                write_json_line(pair, file_)
        with open(path+"validation.json", "w") as file_:
            for pair in validation_pairs:
                write_json_line(pair, file_)
# def make_evaluation_pairs:
#     """ This function makes as many pairs for a
#     company as classes. This data is used for the final
#     evaluation against all classes
#     """



if __name__=="__main__":
    class_descriptions = read_descriptions()
    companies_descriptions= read_meta()
    class_descriptions, companies_descriptions = web_des_intersection(class_descriptions, companies_descriptions)
    #invert companies descriptions dictionairy
    classes_companies = defaultdict(list)
    for id_ in companies_descriptions:
        classes_companies[companies_descriptions[id_]["class_num"]].append(id_)
    folds = make_N_folds_classes_equal_datapoints(class_descriptions, companies_descriptions)
    class_folds = merge_folds(folds)
    make_training_dataset(class_folds, class_descriptions, companies_descriptions, classes_companies)
    # for i, j in data:
    #     print(len(i), len(j))



