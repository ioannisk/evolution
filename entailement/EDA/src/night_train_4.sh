python -u my_evaluate.py /home/ioannis/models/recovery_test/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.noise2 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise2.txt > log42.txt
python -u my_evaluate.py /home/ioannis/models/recovery_test/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.noise4 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise4.txt > log44.txt
python -u my_evaluate.py /home/ioannis/models/recovery_test/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.noise6 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise6.txt > log46.txt
python -u my_evaluate.py /home/ioannis/models/recovery_test/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.noise8 ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_noise8.txt > log48.txt
python -u my_evaluate.py /home/ioannis/models/recovery_test/model4 /home/ioannis/data/recovery_test/fold4/ranking_validation.json.vocab_clean ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_vocab_clean.txt > log4c.txt
