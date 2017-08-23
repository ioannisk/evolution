
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold0/training.json /home/ioannis/data/recovery_test/fold0/validation.json /home/ioannis/models/eda_models/model0 mlp --lower -e 5 -u 220 -d 0.7  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt > log_0.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold1/training.json /home/ioannis/data/recovery_test/fold1/validation.json /home/ioannis/models/eda_models/model1 mlp --lower -e 5 -u 220 -d 0.7  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt > log_1.txt
python -u my_train.py ~/data/glove/glove-840B_l2.npy /home/ioannis/data/recovery_test/fold2/training.json /home/ioannis/data/recovery_test/fold2/validation.json /home/ioannis/models/eda_models/model2 mlp --lower -e 5 -u 220 -d 0.7  -b 32 -r 0.02 --report 20000 --vocab ~/data/glove/glove-840B-vocabulary_l2.txt > log_2.txt







# python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise2 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise2.txt > log32.txt
# python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise4 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise4.txt > log34.txt
# python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise6 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise6.txt > log36.txt
# python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise8 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise8.txt > log38.txt
# python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.vocab_clean ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_vocab_clean.txt > log3c.txt
