import json
import sys
from collections import defaultdict
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
def write_json_line(json_ ,file_):
    json.dump(json_ , file_)
    file_.write('\n')
## This code makes a random split of validation, testing
# ids = []
# des_ids = []
# data= []
# with open("/home/ioannis/data/recovery_test/fold2/ranking_validation.json", "r") as file_:
#     counter = 1
#     for line in file_:
#         line = json.loads(line.strip())
#         data.append(line)
# counter = 0
# valid_subset = open("/home/ioannis/data/recovery_test/fold2/ranking_validation.json_validation_subset", 'w')
# testing_subset = open("/home/ioannis/data/recovery_test/fold2/ranking_validation.json_testing_subset", 'w')
# for i in range(0, len(data), 556):
#     datapoint = data[i:i+556]
#     counter +=1
#     if counter < 1000:
#         for d in datapoint:
#             write_json_line(d, valid_subset)
#     else:
#         for d in datapoint:
#             write_json_line(d, testing_subset)

## This code makes a 1 shot validation
data= []
with open("/home/ioannis/data/recovery_test/fold2/ranking_validation.json", "r") as file_:
    counter = 1
    for line in file_:
        line = json.loads(line.strip())
        data.append(line)
# valid_subset = open("/home/ioannis/data/recovery_test/fold2/ranking_validation.json_valid", 'w')
# testing_subset = open("/home/ioannis/data/recovery_test/fold2/ranking_validation.json_test", 'w')
classes = defaultdict(list)
for i in range(0, len(data), 556):
    datapoint = data[i:i+556]
    web_id = datapoint[0]["web_id"]
    web_class = datapoint[0]["web_class"]
    classes[web_class].append(datapoint)
for cc in classes:
    print(cc, len(classes[cc]))
print(len(classes))



    # datapoint[0][]




