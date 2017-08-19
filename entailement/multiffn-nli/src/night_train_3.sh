python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise2 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise2.txt > log32.txt
python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise4 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise4.txt > log34.txt
python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise6 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise6.txt > log36.txt
python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.noise8 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise8.txt > log38.txt
python -u my_evaluate.py /home/ioannis/models/recovery_test/model3 /home/ioannis/data/recovery_test/fold3/ranking_validation.json.vocab_clean ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_vocab_clean.txt > log3c.txt
