import pickle
import pandas
import json
import random
import time
import os
from attention_evaluation import load_json_validation_file
#### data -> dictionairy -> json


# df_web = pickle.load( open("../data/df_web.pkl","rb"))
# des_df = pickle.load(open("../data/des_df.pkl","rb"))

MAX_LEN=115
MAX_DES_LEN=MAX_LEN
MAX_WEB_LEN=MAX_LEN

def write_json_line(json_ ,file_):
    json.dump(json_ , file_)
    file_.write('\n')

def delete_difference(dic1,dic2):
    a = set(dic1.keys())
    b = set(dic2.keys())
    diff =  a.symmetric_difference(b)
    for key in diff:
        if key in dic1:
            del dic1[key]
        if key in dic2:
            del dic2[key]
    return dic1, dic2

def read_data():
    des = {}
    web_class = {}
    id_txt = {}
    id_class = {}
    max_des = 0
    des_lens = []
    with open("../data/descriptions_data_1.txt","r") as file_:
        for line in file_:
            line = line.strip()
            class_num , txt = line.split('\t')
            if len(txt.split()) <=MAX_DES_LEN:
                des[class_num] = txt
    with open("../data/web_site_meta_1.txt", "r") as file_:
    # with open("../data/web_site_data.txt", "r") as file_:
        for line in file_:
            line = line.strip()
            try:
                id_, class_num , txt = line.split('\t')
                if class_num in set(des.keys()):
                    if len(txt.split()) <=MAX_WEB_LEN:
                        # if max_des <= len(txt.split()):
                        #     max_des = len(txt.split())
                        if class_num in web_class:
                            web_class[class_num].append((id_, txt))
                        else:
                            web_class[class_num] = []
                            web_class[class_num].append((id_, txt))
                        id_txt[id_] = txt
                        id_class[id_] = class_num
            except:
                pass
    print(len(id_txt.keys()))
    des, web_class = delete_difference(des, web_class)
    return des, web_class, id_txt, id_class


def make_pairs(des, web_class, id_txt, id_class):
    positive = []
    negative = []
    ids = set(id_txt.keys())
    classes = set(des.keys())
    for class_num in web_class:
        des_txt = des[class_num]
        for id_, web_txt in web_class[class_num]:
            positive.append({'des':des_txt, 'web':web_txt, 'class':"entailment",
             'des_class':class_num , 'web_class':class_num, 'web_id':id_ })
    file_training =  open("../data/training_pairs_{}.json".format(MAX_LEN), 'wb')
    file_validation =  open("../data/validation_pairs_{}.json".format(MAX_LEN), 'wb')
    counter =0
    for i in positive:
        counter +=1
        if counter >= int(len(ids)*0.95):
            json.dump(i, file_validation)
            file_validation.write('\n')
        else:
            json.dump(i, file_training)
            file_training.write('\n')
    counter = 0
    for i in range(len(positive)):
        counter +=1
        if counter % 1000 ==0:
            print(counter)
        # take one random sample (sample is return as a list so take 0 element)
        id_ = random.sample(ids, 1)[0]
        unsuccesful_sample = True
        while unsuccesful_sample:
            cl = random.sample(classes,1)[0]
            if id_class[id_]==cl:
                unsuccesful_sample = True
            else:
                unsuccesful_sample = False
        negative.append({'des':des[cl], 'web':id_txt[id_], 'class':"contradiction",
            'des_class':cl, 'web_class':id_class[id_], 'web_id':id_ })
    counter = 0
    for i in negative:
        counter +=1
        if counter >= int(len(ids)*0.95):
            json.dump(i, file_validation)
            file_validation.write('\n')
        else:
            json.dump(i, file_training)
            file_training.write('\n')


def merge_lists(file_, outfile):
    en_list = []
    con_list = []
    for line in file_:
        line = line.strip()
        line = json.loads(line)
        en_list.append(line) if line["class"] =="entailment" else con_list.append(line)
    for en, con in zip(en_list, con_list):
        write_json_line(en, outfile)
        write_json_line(con, outfile)



def shuffle_data():
    training_data = []
    validation_data = []
    file_training =  open("../data/training_pairs_{0}.json".format(MAX_LEN), 'rb')
    file_validation =  open("../data/validation_pairs_{0}.json".format(MAX_LEN), 'rb')
    file_training_out =  open("../data/meta_training_{0}.json".format(MAX_LEN), 'wb')
    file_validation_out =  open("../data/meta_validation_{0}.json".format(MAX_LEN), 'wb')
    merge_lists(file_training,file_training_out)
    merge_lists(file_validation,file_validation_out)
    os.remove("../data/training_pairs_{0}.json".format(MAX_LEN))
    os.remove("../data/validation_pairs_{0}.json".format(MAX_LEN))


def find_only_used_classes():
    used_classes = set()
    # count_dic = Counter()
    sum_all = 0.0
    with open("/home/ioannis/evolution/data/meta_training_{}.json".format(MAX_LEN),"r") as file_:
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            used_classes.add(line["web_class"])
            # count_dic[line["web_class"]] +=1
            sum_all +=1
    with open("/home/ioannis/evolution/data/meta_validation_{}.json".format(MAX_LEN),"r") as file_:
        for line in file_:
            line = line.strip()
            line = json.loads(line)
            used_classes.add(line["web_class"])
            # count_dic[line["web_class"]] +=1
            sum_all +=1
    # cmon = count_dic.most_common()
    # for i,j in cmon:
    #     print (i, j*100/sum_all)
    return used_classes

def make_ranking_validation():
    ##########################
    # Load data necessary for making the raning validation data
    #########################
    descriptions_class = []
    descriptions_txt = []
    used_classes = find_only_used_classes()
    with open("../data/meta_validation_{0}.json".format(MAX_LEN), 'r') as file_:
        des_txt, web_txt, binary_class, des_class, web_class, web_id = load_json_validation_file(file_)
    print("Different website txt {0}".format(len(web_id)))
    with open("/home/ioannis/evolution/data/descriptions_data.txt","r") as file_:
        for line in file_:
            line = line.strip()
            line = line.split('\t')
            descriptions_class.append(line[0])
            descriptions_txt.append(line[1])
    ###############
    # Make the actual dataset
    ###############
    with open("/home/ioannis/evolution/data/meta_ranking_validation_{0}.json".format(MAX_LEN),"w") as file_:
        counter = 0
        for i, website_txt in enumerate(web_txt):
            if counter % 100==0:
                print(counter)
            counter +=1
            if binary_class[i]!="entailment":
                continue
            id_ = web_id[i]
            true_cl = des_class[i]
            for description_class, description_txt in zip(descriptions_class, descriptions_txt):
                if description_class==true_cl:
                    class_buffer = 'entailment'
                else:
                    class_buffer = 'contradiction'
                    # json_buffer={'des':description_txt , 'web':website_txt , 'class':'entailment', 'web_id':id_, 'web_class':web_class[i], 'des_class':description_class}
                json_buffer={'des':description_txt , 'web':website_txt , 'class':class_buffer, 'web_id':id_, 'web_class':web_class[i], 'des_class':description_class}
                write_json_line(json_buffer,file_)

if __name__ =="__main__":
    # des, web_class, id_txt, id_class  = read_data()
    # make_pairs(des, web_class, id_txt, id_class)
    # shuffle_data()
    make_ranking_validation()

