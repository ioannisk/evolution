

### Have not run it yet ###
### need to run this tomorrow morning

python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold1/training.json /home/ioannis/data/recovery_test/fold1/validation.json /home/ioannis/models/eda_models_2/model1 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 60000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold2/training.json /home/ioannis/data/recovery_test/fold2/validation.json /home/ioannis/models/eda_models_2/model2 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 60000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold3/training.json /home/ioannis/data/recovery_test/fold3/validation.json /home/ioannis/models/eda_models_2/model3 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 60000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold4/training.json /home/ioannis/data/recovery_test/fold4/validation.json /home/ioannis/models/eda_models_2/model4 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 60000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt


# python -u my_evaluate.py /home/ioannis/models/eda_models_2/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models_2/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models_2/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models_2/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt

# evaluation running atm
