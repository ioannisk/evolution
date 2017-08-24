### Predicitons are re done for eda_models models
### this models are getting trained right now check both of them tomorrow


python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/data/recovery_test/fold1/training.json /home/ioannis/data/recovery_test/fold1/validation.json /home/ioannis/models/eda_models_3/model1 mlp --lower -e 4 -u 220 -d 0.75  -b 32 -r 0.03 --report 10000 --vocab ~/data/glove/glove-840B-vocabulary.txt
python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/data/recovery_test/fold2/training.json /home/ioannis/data/recovery_test/fold2/validation.json /home/ioannis/models/eda_models_3/model2 mlp --lower -e 4 -u 220 -d 0.75  -b 32 -r 0.03 --report 10000 --vocab ~/data/glove/glove-840B-vocabulary.txt
python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/data/recovery_test/fold3/training.json /home/ioannis/data/recovery_test/fold3/validation.json /home/ioannis/models/eda_models_3/model3 mlp --lower -e 4 -u 220 -d 0.75  -b 32 -r 0.03 --report 10000 --vocab ~/data/glove/glove-840B-vocabulary.txt
python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/data/recovery_test/fold4/training.json /home/ioannis/data/recovery_test/fold4/validation.json /home/ioannis/models/eda_models_3/model4 mlp --lower -e 4 -u 220 -d 0.75  -b 32 -r 0.03 --report 10000 --vocab ~/data/glove/glove-840B-vocabulary.txt

# python -u my_evaluate.py /home/ioannis/models/eda_models_3/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models_3/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models_3/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models_3/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt


#### Evaluation has to be done for this :) please do it
