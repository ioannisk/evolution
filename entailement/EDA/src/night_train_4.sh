
# python  my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold3/training.json /home/ioannis/data/recovery_test/fold3/validation.json /home/ioannis/models/eda_models/model3 mlp --lower -e 4 -u 220 -d 0.7  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python  my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold4/training.json /home/ioannis/data/recovery_test/fold4/validation.json /home/ioannis/models/eda_models/model4 mlp --lower -e 4 -u 220 -d 0.7  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt


python  my_evaluate.py /home/ioannis/models/eda_models/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt
python  my_evaluate.py /home/ioannis/models/eda_models/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt
