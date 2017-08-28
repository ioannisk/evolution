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



python -u my_evaluate.py /home/ioannis/models/best_eda/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.filter ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/best_eda/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.filter ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_filter.txt
python -u my_evaluate.py /home/ioannis/models/best_eda/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.filter ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_filter.txt

# ranking_validation.json.filter

#### Evaluation has to be done for this :) please do it
# python -u my_evaluate.py /home/ioannis/models/best_eda/model0 /home/ioannis/data/recovery_test/fold0/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_super.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model3 /home/ioannis/data/recovery_test/fold3/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_super.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model4 /home/ioannis/data/recovery_test/fold4/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_super.txt

