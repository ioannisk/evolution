import pandas
import json
import random
import time
import os
from collections import Counter, defaultdict
from attention_evaluation import load_json_validation_file
import numpy as np

# Not actually 20 folds
# 2k in each bucket is convenient number for testing quickly
N = 3
# MAX_LEN= 120
supervised_validation_volume = 8000
# MAX_DES_LEN=MAX_LEN
# MAX_WEB_LEN=MAX_LEN
data_path = "../data/1rfolds{}_sl_filtered/".format(N)
# data_path = "../data/10rfolds{}/".format(N)
# data_path = "../data/100rfolds{}/".format(N)

def clean_up_txt(page_txt):
    page_txt = page_txt.lower()
    page_txt = re.sub('\s+',' ',page_txt)
    # page_txt = re.sub('[^0-9a-zA-Z]+', " ", page_txt)
    page_txt = re.sub('[^a-zA-Z]+', " ", page_txt)
    return page_txt


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
            #
            # experiment with cutting the max len rather than excuding
            #
            # if len(txt.split()) <=MAX_DES_LEN:
            #     class_descriptions[class_num] = txt
            # print(" ".join(txt.split()[:100]))
            # stop
            class_descriptions[class_num] = " ".join(txt.split()[:100])
            # print(class_descriptions[class_num])
    print(len(class_descriptions))
    return class_descriptions

def read_meta():
    """Read the meta data file
    input: meta txt file produced from wrt_locally.py
    output: {id: {"class_num":, "txt": } }
    """
    companies_descriptions = {}
    with open("/home/ioannis/data/web_sic_description.tsv", "r") as file_:
        counter = 0
        for i, line in enumerate(file_):
            line = line.strip()
            id_ = i
            try:
                class_num , txt = line.split('\t')
                txt = clean_up_txt(txt)
                counter +=1
            except:
                pass
                continue
            if len(txt.split()) <= MAX_WEB_LEN:
                companies_descriptions[id_] = {"class_num":class_num, "txt":txt}
            # txt = " ".join(txt.split()[:100])
            # companies_descriptions[id_] = {"class_num":class_num, "txt":txt}
    print(len(companies_descriptions))
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
    os.mkdir(data_path)
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
    with open(data_path+"volumes_data.txt","w") as file_:
        file_.write("Folds have volume of {}\n".format(list(zip(folds_volume, range(len(folds_volume))))))
        file_.write("Folds have #classes of {}\n".format(list(zip([len(i) for i in folds],range(len(folds_volume))))))
    return folds


def training_validation_split(class_descriptions,companies_descriptions):
    """Desing big experiment, train 3 models with 3 fodls of held out classes
    while we also keep some companies out from the seen classes so we can test SL learning
    """
    os.mkdir(data_path)
    all_classes = list(class_descriptions.keys())
    class_validation_N = 10
    companies_validation = 5000
    splits = 3
    folds = []
    allowed_samples = all_classes
    class_counts = Counter()
    for id_ in companies_descriptions:
        class_counts[companies_descriptions[id_]["class_num"]]+=1
    for split in range(splits):
        folds_samples = []
        volume_fold = 0
        # while len(folds_samples) <class_validation_N:
        while volume_fold < 6000:
            sample = np.random.choice(allowed_samples, 1, replace=False)[0]
            if class_counts[sample] < 3000:
                volume_fold +=class_counts[sample]
                allowed_samples.remove(sample)
                folds_samples.append(sample)
        folds.append(folds_samples)
    print(folds)
    # ranked = class_counts.most_common()[::-1]
    # print(ranked)
    # Rank according to least common count
    ## GOOD LIST PRODUCED FOR SPLIT 3
    folds = [['47990', '38310', '46431', '51220', '17230', '47220', '23520', '27110', '32110', '26701', '25120', '95240'], ['85410', '23640', '20412', '27110', '65202', '46341', '59120', '72200', '24520', '66120', '28301', '82190', '52102', '20302', '13950', '47782', '74203'], ['20150', '71121', '14390', '46210', '63990', '23510', '31030', '51220', '28120', '46380', '81223', '85520']]

    for fold in folds:
        fold_sum = 0
        for class_ in fold:
            fold_sum += class_counts[class_]
        print(fold_sum)

    training_sets = []
    for fold in folds:
        training = [class_ for class_ in list(class_descriptions.keys()) if class_ not in fold]
        training_sets.append(training)
    for i in range(3):
        print(len(folds[i]), len(training_sets[i]), (len(folds[i]) + len(training_sets[i]) ))

    data = list(zip(training_sets, folds))
    return data









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


def make_pairs(fold_classes,class_descriptions, companies_descriptions,classes_companies, TRAINING=False):
    """ Match the document with the
    des class if match or a random
    des class if not match. Then shuffle
    so MLP can learn.
    """

    ## keep out a supervised training set of 10k companies
    if TRAINING:
        supervised_validation_data = []
        companies_in_fold_all_info = [ (company,class_) for class_ in fold_classes for company in classes_companies[class_]]
        random.shuffle(companies_in_fold_all_info)
        supervised_validation_with_classes = companies_in_fold_all_info[:supervised_validation_volume]
        supervised_validation, supervised_validation_classes = zip(*supervised_validation_with_classes)
        supervised_validation = set(supervised_validation)
        for company in supervised_validation:
            website_txt = companies_descriptions[company]["txt"]
            web_class = companies_descriptions[company]["class_num"]
            id_ = company
            for class_num in class_descriptions:
                class_buffer ='entailment' if class_num ==web_class else 'contradiction'
                json_buffer={'des':class_descriptions[class_num] , 'web':website_txt ,
                'class':class_buffer, 'web_id':id_, 'web_class':web_class, 'des_class':class_num}
                supervised_validation_data.append(json_buffer)

    positive = []
    negative = []
    ## positive pairs
    print(len(fold_classes))
    for class_ in fold_classes:

        companies = classes_companies[class_]
        class_des = class_descriptions[class_]
        for company in companies:
            if TRAINING:
                if company in supervised_validation:
                    continue
            company_des = companies_descriptions[company]["txt"]
            company_class = companies_descriptions[company]["class_num"]
            json_buffer = {'des':class_des, 'web':company_des, 'class':"entailment",
            'des_class':class_, 'web_class':company_class, 'web_id':company}
            positive.append(json_buffer)

    # print(len(supervised_validation))
    # print(len(companies_in_fold_all_info))
    # print(len(set(supervised_validation_classes)))
    # stop
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
            if TRAINING:
                if company in supervised_validation:
                    continue
            company_des = companies_descriptions[company]["txt"]
            company_class = companies_descriptions[company]["class_num"]
            sample_class = allowed_samples[random.randint(0, len(allowed_samples) - 1)]
            class_des = class_descriptions[sample_class]
            json_buffer = {'des':class_des, 'web':company_des, 'class':"contradiction",
            'des_class':sample_class, 'web_class':company_class, 'web_id':company}
            negative.append(json_buffer)
    ## shuffle data for learning purpose
    # print("########")
    # print(len(companies_descriptions.keys()))
    # print(len(supervised_validation))
    # print("########")
    # stop

    data = positive + negative
    random.shuffle(data)
    if TRAINING:
        return data, supervised_validation_data
    else:
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

    # try:
    # os.mkdir(data_path)
    # except:
    #     pass
    for i, (training, validation) in enumerate(class_folds):
        print("Writting fold {}".format(i))
        training_pairs, supervised_validation_data = make_pairs(training,class_descriptions, companies_descriptions,classes_companies, TRAINING = True)
        validation_pairs = make_pairs(validation,class_descriptions, companies_descriptions,classes_companies)
        path = data_path + "fold{}/".format(i)
        try:
            os.mkdir(path)
        except:
            pass
        with open(path+"training.json", "w") as file_:
            print("writing trainign set")
            for pair in training_pairs:
                write_json_line(pair, file_)
        with open(path+"validation.json", "w") as file_:
            print("writing validation set")
            for pair in validation_pairs:
                write_json_line(pair, file_)

        with open(path+"supervised_validation.json", "w") as file_:
            print("writing supervised_validation set")
            for pair in supervised_validation_data:
                write_json_line(pair, file_)
        # with open()
    return



def make_evaluation_pairs(class_descriptions):
    """ This function makes as many pairs for a
    company as classes. This data is used for the final
    evaluation against all classes
    """
    folds = os.listdir(data_path)
    for fold in folds:
        if not os.path.isdir(data_path+fold):
            continue
        print("Writing:  {}".format(fold))
        fold_path = data_path+fold
        json_files =os.listdir(fold_path)
        for file_ in json_files:
            if file_!="validation.json":
                continue
            file_path = data_path+fold+"/"+file_
            with open(file_path, 'r') as validation_file:
                # This function returns only entailements
                des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(validation_file)
            ranking_file=data_path+fold+"/ranking_validation.json"
            with open(ranking_file, 'w') as ranking_validation:
                for i, website_txt in enumerate(web_txt):
                    id_ = web_id[i]
                    true_cl = des_class[i]
                    for class_num in class_descriptions:
                        class_buffer ='entailment' if class_num ==true_cl else 'contradiction'
                        json_buffer={'des':class_descriptions[class_num] , 'web':website_txt , 'class':class_buffer, 'web_id':id_, 'web_class':web_class[i], 'des_class':class_num}
                        write_json_line(json_buffer,ranking_validation)


# def make_supervised_evaluation():



if __name__=="__main__":
    class_descriptions = read_descriptions()
    companies_descriptions= read_meta()
    class_descriptions, companies_descriptions = web_des_intersection(class_descriptions, companies_descriptions)
    print(len(class_descriptions), len(companies_descriptions))
    stop
    # # #invert companies descriptions dictionairy
    classes_companies = defaultdict(list)
    for id_ in companies_descriptions:
        classes_companies[companies_descriptions[id_]["class_num"]].append(id_)
    class_folds = training_validation_split(class_descriptions,companies_descriptions)
    supervised_validations = make_training_dataset(class_folds, class_descriptions, companies_descriptions, classes_companies)
    make_evaluation_pairs(class_descriptions)

