
# # OLD CONTENT
# # python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold0/training.json /home/ioannis/data/recovery_test/fold0/validation.json /home/ioannis/models/eda_models/model0 mlp --lower -e 7 -u 200 -d 0.8  -b 32 -r 0.05 --report 200000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_rep.txt



# # python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold0/training.json /home/ioannis/data/recovery_test/fold0/validation.json /home/ioannis/models/eda_models_1/model0 mlp --lower -e 7 -u 200 -d 0.8  -b 32 -r 0.05 --report 200000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt
# python -u my_evaluate.py /home/ioannis/models/eda_models_1/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_rep.txt

# python -u my_evaluate.py /home/ioannis/models/best_eda/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.vocab_clean ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_vocab_clean.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.vocab_clean ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_vocab_clean.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.vocab_clean ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_vocab_clean.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.vocab_clean ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_vocab_clean.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.vocab_clean ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_vocab_clean.txt


# python -u my_evaluate.py /home/ioannis/models/best_eda/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.noise2 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise_2.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.noise2 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise_2.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model2 /home/ioannis/data/recovery_test/fold2/ranking_validation.json.noise2 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise_2.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise2 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise_2.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.noise2 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise_2.txt

# python -u my_evaluate.py /home/ioannis/models/best_eda/model0 /home/ioannis/data/recovery_test/fold0/ranking_validation.json.filter ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_filter.txt
# python -u my_evaluate.py /home/ioannis/models/best_eda/model1 /home/ioannis/data/recovery_test/fold1/ranking_validation.json.filter ~/data/glove/glove-840B_l2.npy --vocabulary ~/data/glove/glove-840B-vocabulary_l2.txt -save prob_predictions_filter.txt


