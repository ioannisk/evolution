import numpy as np

models = ["eda_models", "eda_models_1"]
folds = [0,1,2,3,4]

for fold in folds:
    print("ensemble {}".format(fold))
    num_lines = sum(1 for line in open('/home/ioannis/models/eda_models/model{}/prob_predictions.txt'.format(fold), 'r'))
    store_pred = np.zeros(num_lines)
    for model in models:
        print("    reading model {}".format(model)))
        with open('/home/ioannis/models/{}/model{}/prob_predictions.txt'.format(model, fold), 'r') as file_:
            for i, line in enumerate(file_):
                line=float(line)
                store_pred[i] += line
    print("    Writting")
    store_pred = store_pred/len(models)
    output = open('/home/ioannis/models/ensemble/model{}/prob_predictions.txt'.format(fold), 'w')
    for i in store_pred:
        output.write("{}\n".format(i))

