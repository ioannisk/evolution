### Predicitons are re done for eda_models models
### this models are getting trained right now check both of them tomorrow

# python -u my_evaluate.py /home/ioannis/models/best_eda/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.noise4 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise4.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.noise4 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise4.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.noise4 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise4.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise4 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise4.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.noise4 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise4.txt

# python -u my_evaluate.py /home/ioannis/models/best_eda/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.noise6 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise6.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.noise6 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise6.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.noise6 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise6.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise6 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise6.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.noise6 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise6.txt

# python -u my_evaluate.py /home/ioannis/models/best_eda/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.noise8 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise8.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.noise8 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise8.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.noise8 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise8.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise8 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise8.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.noise8 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise8.txt



# python -u my_evaluate.py /home/ioannis/models/best_eda/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.filter ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_filter.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.filter ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_filter.txt

# ranking_validation.json.filter

#### running atm
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold0/training.json.filter /home/ioannis/data/recovery_test/fold0/validation.json /home/ioannis/models/filtered_models_1/model0 mlp --lower -e 3 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold1/training.json.filter /home/ioannis/data/recovery_test/fold1/validation.json /home/ioannis/models/filtered_models_1/model1 mlp --lower -e 3 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold2/training.json.filter /home/ioannis/data/recovery_test/fold2/validation.json /home/ioannis/models/filtered_models_1/model2 mlp --lower -e 3 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold3/training.json.filter /home/ioannis/data/recovery_test/fold3/validation.json /home/ioannis/models/filtered_models_1/model3 mlp --lower -e 3 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold4/training.json.filter /home/ioannis/data/recovery_test/fold4/validation.json /home/ioannis/models/filtered_models_1/model4 mlp --lower -e 3 -u 200 -d 0.8  -b 32 -r 0.05 --report 130000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt

python -u my_evaluate.py /home/ioannis/models/filtered_models_1/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/filtered_models_1/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/filtered_models_1/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/filtered_models_1/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/filtered_models_1/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
