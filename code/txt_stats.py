import json
import sys
vocab = set()
# with open("/home/ioannis/data/snli_1.0/snli_1.0_train.jsonl", "r") as file_:
#     for line in file_:
#         line = json.loads(line.strip())
#         sen1 = line['sentence1']
#         sen2 = line['sentence2']
#         data = sen1 + " "+ sen2
#         for word in data.split():
#             vocab.add(word)
# print(len(vocab))
ids = []
des_ids = []
with open("/home/ioannis/data/recovery_test/fold0/ranking_validation.json", "r") as file_:
    counter = 1
    for line in file_:
        line = json.loads(line.strip())
        web_id = line["web_id"]
        # print("-------")
        # print(counter)
        # print(web_id)

        des_class = line["des_class"]
        ids.append(web_id)
        des_ids.append(des_class)
        # if counter == 556:
        #     # print(ids)
        #     # print(len(des_ids))
        #     ids = set()
        #     des_ids = set()
        #     counter = 0
        counter +=1

import IPython; IPython.embed()
# print(len(ids))
# print(len(des_ids))
