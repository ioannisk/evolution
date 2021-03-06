import numpy as np
from attention_evaluation import load_json_validation_file
import json

# choosen_fold = "1rfolds3"
# data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)
# for li
# for i in range(3):
# i = 0
# with open(data_path+"fold{}/validation.json".format(i),"r") as file_:
#     print("Analysing fold {}".format(i))
#     counter = 0
#     used_classes = set()
#     for line in file_:
#         counter +=1
#         if counter % 100000 == 0:
#             print(counter)
#         line = line.strip()
#         line = json.loads(line)
#         # des = line["des"]
#         # web = line["web"]
#         class_= line["web_id"]
#         used_classes.add(class_)
#         # if len(des.split())==0 or len(web.split())==0:
#         #     print(line[web_id])
#     # print(used_classes)

# choosen_fold = "1rfolds3_sl"
# data_path = "/home/ioannis/evolution/data/{}/".format(choosen_fold)
# with open(data_path+"fold{}/validation.json".format(i),"r") as file_:
#     print("Analysing fold {}".format(i))
#     counter = 0
#     used_classes_1 = set()
#     for line in file_:
#         counter +=1
#         if counter % 100000 == 0:
#             print(counter)
#         line = line.strip()
#         line = json.loads(line)
#         # des = line["des"]
#         # web = line["web"]
#         class_= line["web_id"]
#         used_classes_1.add(class_)
#         # if len(des.split())==0 or len(web.split())==0:
#         #     print(line[web_id])
# print(used_classes_1==used_classes)
for k in  [1,2,3,4]:
    # rank_valid = open("/home/ioannis/data/recovery_test/fold{}/ranking_validation.json.filter".format(k), 'r')
    valid = open("/home/ioannis/data/recovery_test/fold{}/validation.json".format(k), 'r')
    train = open("/home/ioannis/data/recovery_test/fold{}/training.json.filter".format(k), 'r')
    training_cl = set()
    for line in train:
        line = json.loads(line.strip())
        if line["class"]=="entailment":
            training_cl.add(line["web_class"])
    valid_cl = set()
    for line in valid:
        line = json.loads(line.strip())
        if line["class"]=="entailment":
            valid_cl.add(line["web_class"])
    # rank_valid_cl = set()
    # for line in rank_valid:
    #     line = json.loads(line.strip())
    #     if line["class"]=="entailment":
    #         rank_valid_cl.add(line["web_class"])
    print(k, len(training_cl), len(valid_cl))

    # for ii in rank_valid_cl:
    #     if ii in training_cl:
    #         print(ii)

# import IPython; IPython.embed()




