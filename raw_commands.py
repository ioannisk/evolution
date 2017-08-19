
###
### NEW FOLDS EXPERIMENT
###

# Train

#### MODEL 1 is a very good candidate very very good perfomance
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/new_data_3/fold0/training.json /home/ioannis/evolution/data/new_data_3/fold0/validation.json new_data_3/model0 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_0.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/new_data_3/fold1/training.json /home/ioannis/evolution/data/new_data_3/fold1/validation.json new_data_3/model1 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_1.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/new_data_3/fold2/training.json /home/ioannis/evolution/data/new_data_3/fold2/validation.json new_data_3/model2 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_2.txt &
#
# Evaluate
#
# nohup python -u my_evaluate.py new_data_3/model0 /home/ioannis/evolution/data/new_data_3/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt > log0.txt &
# nohup python -u my_evaluate.py new_data_3/model1 /home/ioannis/evolution/data/new_data_3/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt > log1.txt &
# nohup python -u my_evaluate.py new_data_3/model2 /home/ioannis/evolution/data/new_data_3/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt > log2.txt &



## Experiment 2
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/new_data_3/fold0/training.json /home/ioannis/evolution/data/new_data_3/fold0/validation.json new_data_3_1/model0 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.02 --report 10000 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_0.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/new_data_3/fold1/training.json /home/ioannis/evolution/data/new_data_3/fold1/validation.json new_data_3_1/model1 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.02 --report 10000 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_1.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/new_data_3/fold2/training.json /home/ioannis/evolution/data/new_data_3/fold2/validation.json new_data_3_1/model2 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.02 --report 10000 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_2.txt &
#
# Evaluate
#
# nohup python -u my_evaluate.py new_data_3_1/model0 /home/ioannis/evolution/data/new_data_3/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt > log0.txt &
# nohup python -u my_evaluate.py new_data_3_1/model1 /home/ioannis/evolution/data/new_data_3/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt > log1.txt &
# nohup python -u my_evaluate.py new_data_3_1/model2 /home/ioannis/evolution/data/new_data_3/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt > log2.txt &
