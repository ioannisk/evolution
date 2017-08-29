
## old content
# nohup python -u my_evaluate.py /home/ioannis/models/eda_models_2/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions.txt &


##### NEW CONTENT
#### Verify that you know how to replicate the predictions: It is important for the noisy predictions
# need to be run
##python -u my_evaluate.py /home/ioannis/models/reproduced/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_rep.txt


#### RUN LAST NIGHT their results should be also tested last night
# python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold0/training.json.filter /home/ioannis/data/recovery_test/fold0/validation.json /home/ioannis/models/filtered_models/model0 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold1/training.json.filter /home/ioannis/data/recovery_test/fold1/validation.json /home/ioannis/models/filtered_models/model1 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold2/training.json.filter /home/ioannis/data/recovery_test/fold2/validation.json /home/ioannis/models/filtered_models/model2 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold3/training.json.filter /home/ioannis/data/recovery_test/fold3/validation.json /home/ioannis/models/filtered_models/model3 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold4/training.json.filter /home/ioannis/data/recovery_test/fold4/validation.json /home/ioannis/models/filtered_models/model4 mlp --lower -e 4 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt


# python -u my_evaluate.py /home/ioannis/models/filtered_models/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
# python -u my_evaluate.py /home/ioannis/models/filtered_models/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
# python -u my_evaluate.py /home/ioannis/models/filtered_models/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
# python -u my_evaluate.py /home/ioannis/models/filtered_models/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
# python -u my_evaluate.py /home/ioannis/models/filtered_models/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt



python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold0/training.json.filter /home/ioannis/data/recovery_test/fold0/validation.json /home/ioannis/models/filtered_models_2/model0 mlp --lower -e 2 -u 200 -d 0.8  -b 32 -r 0.05 --report 70000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold1/training.json.filter /home/ioannis/data/recovery_test/fold1/validation.json /home/ioannis/models/filtered_models_2/model1 mlp --lower -e 2 -u 200 -d 0.8  -b 32 -r 0.05 --report 70000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold2/training.json.filter /home/ioannis/data/recovery_test/fold2/validation.json /home/ioannis/models/filtered_models_2/model2 mlp --lower -e 2 -u 200 -d 0.8  -b 32 -r 0.05 --report 70000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold3/training.json.filter /home/ioannis/data/recovery_test/fold3/validation.json /home/ioannis/models/filtered_models_2/model3 mlp --lower -e 2 -u 200 -d 0.8  -b 32 -r 0.05 --report 70000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold4/training.json.filter /home/ioannis/data/recovery_test/fold4/validation.json /home/ioannis/models/filtered_models_2/model4 mlp --lower -e 2 -u 200 -d 0.8  -b 32 -r 0.05 --report 70000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt

python -u my_evaluate.py /home/ioannis/models/filtered_models_2/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/filtered_models_2/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/filtered_models_2/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/filtered_models_2/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/filtered_models_2/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt


