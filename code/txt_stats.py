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
CHOOSE_MODEL = "best_eda"
# CHOOSE_MODEL = "filtered_models_2"
CHOOSEN_FOLD = 4
fold = {0:[61200,22110,14132,21200,52101,58210],
        1:[46341,13950,74203,20412,52102,82190],
        2:[28120,81223,31030,14390,20150],
        3:[28410,18130,10840,28120,46360,26514],
        4:[23320,65300,46210,13910,10831,63910]
}
with open("/home/ioannis/models/{}/model{}/prob_predictions_filter.txt".format(CHOOSE_MODEL,CHOOSEN_FOLD), "r") as file_:
    predictions = []
    for line in file_:
        line = line.strip()
        predictions.append(float(line))
data = []
with open("/home/ioannis/data/recovery_test/fold{}/ranking_validation.json.filter".format(CHOOSEN_FOLD), "r") as file_:
    counter = 1
    for line in file_:
        line = json.loads(line.strip())
        data.append(line)
# valid_subset = open("/home/ioannis/data/recovery_test/fold{}/ranking_validation.json_valid".format(CHOOSEN_FOLD), 'w')
valid_subset_pred = open("/home/ioannis/models/{}/model{}/prob_predictions_valid.txt".format(CHOOSE_MODEL,CHOOSEN_FOLD), 'w')

# testing_subset = open("/home/ioannis/data/recovery_test/fold{}/ranking_validation.json_test".format(CHOOSEN_FOLD), 'w')
testing_subset_pred = open("/home/ioannis/models/{}/model{}/prob_predictions_test.txt".format(CHOOSE_MODEL,CHOOSEN_FOLD), 'w')
classes = defaultdict(list)
for i in range(0, len(data), 556):
    datapoint = data[i:i+556]
    datapoint_pred = predictions[i:i+556]
    web_id = datapoint[0]["web_id"]
    web_class = datapoint[0]["web_class"]
    classes[web_class].append(zip(datapoint, datapoint_pred))
# for cl in classes:
#     print(cl, len(classes[cl]))
# fold2 = [28120,81223,31030,14390,20150]
# fold1 = [46341,13950,74203,20412,52102,82190]
for cl in classes:
    if int(cl) in fold[CHOOSEN_FOLD]:
        for datapoints in classes[cl]:
            for d, pred in datapoints:
                # write_json_line(d, valid_subset)
                valid_subset_pred.write("{}\n".format(pred))
    else:
        for datapoints in classes[cl]:
            for d, pred in datapoints:
                # write_json_line(d, testing_subset)
                testing_subset_pred.write("{}\n".format(pred))


# fold0 =
# fold1 =
# fold2 =
# fold3 =
# fold4 =




