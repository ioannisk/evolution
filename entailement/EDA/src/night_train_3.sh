

### Have not run it yet ###
### need to run this tomorrow morning
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold0/training.json /home/ioannis/data/recovery_test/fold0/validation.json /home/ioannis/models/eda_models_2/model0 mlp --lower -e 6 -u 190 -d 0.75  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold1/training.json /home/ioannis/data/recovery_test/fold1/validation.json /home/ioannis/models/eda_models_2/model1 mlp --lower -e 6 -u 190 -d 0.75  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold2/training.json /home/ioannis/data/recovery_test/fold2/validation.json /home/ioannis/models/eda_models_2/model2 mlp --lower -e 6 -u 190 -d 0.75  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold3/training.json /home/ioannis/data/recovery_test/fold3/validation.json /home/ioannis/models/eda_models_2/model3 mlp --lower -e 6 -u 190 -d 0.75  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold4/training.json /home/ioannis/data/recovery_test/fold4/validation.json /home/ioannis/models/eda_models_2/model4 mlp --lower -e 6 -u 190 -d 0.75  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
