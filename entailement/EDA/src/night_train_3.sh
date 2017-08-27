

# nohup python -u my_evaluate.py /home/ioannis/models/eda_models_2/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt &


#### Verify that you know how to replicate the predictions: It is important for the noisy predictions



python -u my_evaluate.py /home/ioannis/models/reproduced/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_rep.txt

# nohup python -u my_evaluate.py /home/ioannis/models/eda_models/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt > log1.txt &
# nohup python -u my_evaluate.py /home/ioannis/models/eda_models/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt > log2.txt &
# nohup python -u my_evaluate.py /home/ioannis/models/eda_models/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt > log3.txt &
# nohup python -u my_evaluate.py /home/ioannis/models/eda_models/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt > log4.txt &
