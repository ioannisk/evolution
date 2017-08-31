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



fold = {0:[61200,22110,14132,21200,52101,58210],
        1:[46341,13950,74203,20412,52102,82190],
        2:[28120,81223,31030,14390,20150],
        3:[28410,18130,10840,28120,46360,26514],
        4:[23320,65300,46210,13910,10831,63910]}

## This code makes a 1 shot validation
# CHOOSE_MODEL = "best_eda"
CHOOSE_MODEL = "filtered_models_3"
for i_fold in [0,1,2,3,4]:
    CHOOSEN_FOLD = i_fold
    print(CHOOSEN_FOLD)
    # with open("/home/ioannis/models/{}/model{}/prob_predictions_filter.txt".format(CHOOSE_MODEL,CHOOSEN_FOLD), "r") as file_:
    #     predictions = []
    #     for line in file_:
    #         line = line.strip()
    #         predictions.append(float(line))
    data = []
    # with open("/home/ioannis/data/recovery_test/fold{}/ranking_validation.json.filter".format(CHOOSEN_FOLD), "r") as file_:
    with open("/home/ioannis/data/recovery_test/fold{}/validation.json".format(CHOOSEN_FOLD), "r") as file_:
        counter = 1
        for line in file_:
            line = json.loads(line.strip())
            data.append(line)
    # valid_subset = open("/home/ioannis/data/recovery_test/fold{}/ranking_validation.json_valid".format(CHOOSEN_FOLD), 'w')
    # valid_subset_pred = open("/home/ioannis/models/{}/model{}/prob_predictions_valid.txt".format(CHOOSE_MODEL,CHOOSEN_FOLD), 'w')

    # testing_subset = open("/home/ioannis/data/recovery_test/fold{}/ranking_validation.json_test".format(CHOOSEN_FOLD), 'w')
    # testing_subset_pred = open("/home/ioannis/models/{}/model{}/prob_predictions_test.txt".format(CHOOSE_MODEL,CHOOSEN_FOLD), 'w')
    classes = defaultdict(list)
    for i in range(0, len(data), 556):
        datapoint = data[i:i+556]
        # datapoint_pred = predictions[i:i+556]
        datapoint_pred = range(0,556)
        web_id = datapoint[0]["web_id"]
        web_class = datapoint[0]["web_class"]
        classes[web_class].append(zip(datapoint, datapoint_pred))
    for cl in classes:
        print(cl, len(classes[cl]))
    # fold2 = [28120,81223,31030,14390,20150]
    # fold1 = [46341,13950,74203,20412,52102,82190]
    # for cl in classes:
    #     if int(cl) in fold[CHOOSEN_FOLD]:
    #         for datapoints in classes[cl]:
    #             for d, pred in datapoints:
    #                 # write_json_line(d, valid_subset)
    #                 valid_subset_pred.write("{}\n".format(pred))
    #     else:
    #         for datapoints in classes[cl]:
    #             for d, pred in datapoints:
    #                 # write_json_line(d, testing_subset)
    #                 testing_subset_pred.write("{}\n".format(pred))


# fold0 =
# fold1 =
# fold2 =
# fold3 =
# fold4 =

top_15_results = {3: [21.909   ,32.733   ,36.744   ,40.720   ,43.707   ,46.693   ,49.358   ,51.498   ,53.590   ,55.248   ,56.954   ,58.598   ,60.150   ,61.424],
4: [23.128   ,32.660   ,37.950   ,41.593   ,45.189   ,48.255   ,51.072   ,53.731   ,56.061   ,57.661   ,59.314   ,60.995   ,62.260   ,63.350],
5: [20.426   ,31.181   ,36.573   ,40.902   ,44.428   ,47.312   ,50.131   ,52.686   ,54.566   ,56.412   ,57.868   ,59.170   ,60.470   ,61.591],
6: [24.628   ,34.710   ,39.734   ,43.301   ,46.302   ,49.066   ,51.389   ,53.411   ,55.314   ,56.791   ,58.410   ,59.799   ,60.910   ,62.073],
7: [22.933   ,34.401   ,40.308   ,44.083   ,46.749   ,49.419   ,51.965   ,54.051   ,55.785   ,57.461   ,59.073   ,60.470   ,62.034   ,63.265],
8:[21.798   ,34.137   ,39.495   ,43.242   ,46.306   ,48.875   ,51.014   ,53.472   ,55.400   ,57.050   ,58.899   ,60.570   ,61.789   ,63.034 ],
9: [23.013   ,33.964   ,38.750   ,42.566   ,45.765   ,48.243   ,50.718   ,53.059   ,55.310   ,57.523   ,59.487   ,60.866   ,62.195   ,63.482],
10 :[23.454   ,33.937   ,38.560   ,42.161   ,45.273   ,48.020   ,50.294   ,52.394   ,54.212   ,55.974   ,57.194   ,58.503   ,59.887   ,61.202]
}

nb_results = [15.149,20.423,23.347,25.922,27.759,29.057,30.335,31.187,31.995,32.843,33.516,34.179,34.693,35.138]


tfidf_results = [19.218   ,25.237   ,29.284   ,32.097   ,34.284   ,36.114   ,37.669   ,39.023   ,40.079   ,40.961   ,41.824   ,42.602   ,43.255   ,43.893 ]
