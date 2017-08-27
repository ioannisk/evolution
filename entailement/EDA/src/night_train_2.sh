
# OLD CONTENT
# python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold0/training.json /home/ioannis/data/recovery_test/fold0/validation.json /home/ioannis/models/eda_models/model0 mlp --lower -e 5 -u 200 -d 0.8  -b 32 -r 0.02 --report 60000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold1/training.json /home/ioannis/data/recovery_test/fold1/validation.json /home/ioannis/models/eda_models/model1 mlp --lower -e 5 -u 200 -d 0.8  -b 32 -r 0.02 --report 60000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt
# evaluation running atm


python -u my_evaluate.py /home/ioannis/models/reproduced/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_rep.txt
python -u my_evaluate.py /home/ioannis/models/reproduced/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_rep.txt
