# -*- coding: utf-8 -*-


# python interactive-eval.py ~/models/snli_trained/ ~/data/glove/glove-840B.npy --vocab ~/data/glove/glove-840B-vocabulary.txt -i -a

### WOW THIS IS SICK VISUALIZATION
# python interactive-eval.py 1rfolds3_2/model1 ~/data/glove/glove-840B.npy --vocab ~/data/glove/glove-840B-vocabulary.txt -i -a
# python my_evaluate.py 1rfolds3_sl_1/model2 //home/ioannis/data/testing_binary_snli_snli.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt


################################################################################################
################################################################################################
################################################################################################
#
# 1rfolds3 directory of data used for this experiments
#
################################################################################################
################################################################################################
################################################################################################
### training experimet #1
## done predictions
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold0/training.json /home/ioannis/evolution/data/1rfolds3/fold0/validation.json 1rfolds3/model0 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_0.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold1/training.json /home/ioannis/evolution/data/1rfolds3/fold1/validation.json 1rfolds3/model1 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_1.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold2/training.json /home/ioannis/evolution/data/1rfolds3/fold2/validation.json 1rfolds3/model2 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_2.txt &

# python my_evaluate.py 1rfolds3/model0 /home/ioannis/evolution/data/1rfolds3/fold0/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py 1rfolds3/model1 /home/ioannis/evolution/data/1rfolds3/fold1/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py 1rfolds3/model2 /home/ioannis/evolution/data/1rfolds3/fold2/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt

# python my_evaluate.py 1rfolds3/model0 /home/ioannis/evolution/data/1rfolds3/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py 1rfolds3/model1 /home/ioannis/evolution/data/1rfolds3/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py 1rfolds3/model2 /home/ioannis/evolution/data/1rfolds3/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt


### training experimet #2
## done predictions
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold0/training.json /home/ioannis/evolution/data/1rfolds3/fold0/validation.json 1rfolds3_1/model0 mlp --lower -e 30 -u 200 -d 0.75  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_0_1.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold1/training.json /home/ioannis/evolution/data/1rfolds3/fold1/validation.json 1rfolds3_1/model1 mlp --lower -e 30 -u 200 -d 0.75  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_1_1.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold2/training.json /home/ioannis/evolution/data/1rfolds3/fold2/validation.json 1rfolds3_1/model2 mlp --lower -e 30 -u 200 -d 0.75  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_2_1.txt &

# nohup python -u my_evaluate.py 1rfolds3_1/model0 /home/ioannis/evolution/data/1rfolds3/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# nohup python -u my_evaluate.py 1rfolds3_1/model1 /home/ioannis/evolution/data/1rfolds3/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# nohup python -u my_evaluate.py 1rfolds3_1/model2 /home/ioannis/evolution/data/1rfolds3/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt



### training experimet #3
#done predictions
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold0/training.json /home/ioannis/evolution/data/1rfolds3/fold0/validation.json 1rfolds3_2/model0 mlp --lower -e 30 -u 250 -d 0.8  -b 32 -r 0.01 --report 350 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_0_2.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold1/training.json /home/ioannis/evolution/data/1rfolds3/fold1/validation.json 1rfolds3_2/model1 mlp --lower -e 30 -u 250 -d 0.8  -b 32 -r 0.01 --report 350 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_1_2.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold2/training.json /home/ioannis/evolution/data/1rfolds3/fold2/validation.json 1rfolds3_2/model2 mlp --lower -e 30 -u 250 -d 0.8  -b 32 -r 0.01 --report 350 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_2_2.txt &


#  python -u my_evaluate.py 1rfolds3_2/model0 /home/ioannis/evolution/data/1rfolds3/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
#  python -u my_evaluate.py 1rfolds3_2/model1 /home/ioannis/evolution/data/1rfolds3/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
#  python -u my_evaluate.py 1rfolds3_2/model2 /home/ioannis/evolution/data/1rfolds3/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt


### training experimet #4
#done predictions
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold0/training.json /home/ioannis/evolution/data/1rfolds3/fold0/validation.json 1rfolds3_3/model0 mlp --lower -e 30 -u 300 -d 0.8  -b 64 -r 0.08 --report 350 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_0_3.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold1/training.json /home/ioannis/evolution/data/1rfolds3/fold1/validation.json 1rfolds3_3/model1 mlp --lower -e 30 -u 300 -d 0.8  -b 64 -r 0.08 --report 350 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_1_3.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3/fold2/training.json /home/ioannis/evolution/data/1rfolds3/fold2/validation.json 1rfolds3_3/model2 mlp --lower -e 30 -u 300 -d 0.8  -b 64 -r 0.08 --report 350 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output_2_3.txt &

#  python -u my_evaluate.py 1rfolds3_3/model0 /home/ioannis/evolution/data/1rfolds3/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
#  python -u my_evaluate.py 1rfolds3_3/model1 /home/ioannis/evolution/data/1rfolds3/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
#  python -u my_evaluate.py 1rfolds3_3/model2 /home/ioannis/evolution/data/1rfolds3/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt

################################################################################################
################################################################################################
################################################################################################
#
# 1rfolds3_sl directory of data used for this experiments
#
################################################################################################
################################################################################################
################################################################################################

# Training experiment 1

# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold0/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold0/validation.json 1rfolds3_sl/model0 mlp --lower -e 30 -u 250 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold1/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold1/validation.json 1rfolds3_sl/model1 mlp --lower -e 30 -u 250 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output1.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold2/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold2/validation.json 1rfolds3_sl/model2 mlp --lower -e 30 -u 250 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output2.txt &



# nohup python -u my_evaluate.py 1rfolds3_sl/model0 /home/ioannis/evolution/data/1rfolds3_sl/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt &
# nohup python -u my_evaluate.py 1rfolds3_sl/model1 /home/ioannis/evolution/data/1rfolds3_sl/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt  > Output1.txt &
# nohup python -u my_evaluate.py 1rfolds3_sl/model2 /home/ioannis/evolution/data/1rfolds3_sl/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt > Output2.txt &


# experiment 2
#### PREDICTIONS HAVE BEEN DONE ALREADY CAREFUL
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold0/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold0/validation.json 1rfolds3_sl_1/model0 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.02 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold1/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold1/validation.json 1rfolds3_sl_1/model1 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.02 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output1.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold2/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold2/validation.json 1rfolds3_sl_1/model2 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.02 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output2.txt &


# python -u my_evaluate.py 1rfolds3_sl_1/model0 /home/ioannis/evolution/data/1rfolds3_sl/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python -u my_evaluate.py 1rfolds3_sl_1/model1 /home/ioannis/evolution/data/1rfolds3_sl/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python -u my_evaluate.py 1rfolds3_sl_1/model2 /home/ioannis/evolution/data/1rfolds3_sl/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt

# nohup python -u my_evaluate.py 1rfolds3_sl_1/model0 /home/ioannis/evolution/data/1rfolds3_sl/fold0/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save supervised_predictions.txt &
# nohup python -u my_evaluate.py 1rfolds3_sl_1/model1 /home/ioannis/evolution/data/1rfolds3_sl/fold1/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save supervised_predictions.txt  &
# nohup python -u my_evaluate.py 1rfolds3_sl_1/model2 /home/ioannis/evolution/data/1rfolds3_sl/fold2/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save supervised_predictions.txt &



# experiment 3
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold0/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold0/validation.json 1rfolds3_sl_2/model0 mlp --lower -e 30 -u 300 -d 0.7  -b 32 -r 0.01 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold1/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold1/validation.json 1rfolds3_sl_2/model1 mlp --lower -e 30 -u 300 -d 0.7  -b 32 -r 0.01 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output1.txt &
# nohup python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/1rfolds3_sl/fold2/training.json /home/ioannis/evolution/data/1rfolds3_sl/fold2/validation.json 1rfolds3_sl_2/model2 mlp --lower -e 30 -u 300 -d 0.7  -b 32 -r 0.01 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt > Output2.txt &

# nohup python -u my_evaluate.py 1rfolds3_sl_2/model0 /home/ioannis/evolution/data/1rfolds3_sl/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt &
# nohup python -u my_evaluate.py 1rfolds3_sl_2/model1 /home/ioannis/evolution/data/1rfolds3_sl/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt  &
# nohup python -u my_evaluate.py 1rfolds3_sl_2/model2 /home/ioannis/evolution/data/1rfolds3_sl/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt &

# nohup python -u my_evaluate.py 1rfolds3_sl_2/model0 /home/ioannis/evolution/data/1rfolds3_sl/fold0/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save supervised_predictions.txt &
# nohup python -u my_evaluate.py 1rfolds3_sl_2/model1 /home/ioannis/evolution/data/1rfolds3_sl/fold1/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save supervised_predictions.txt  &
# nohup python -u my_evaluate.py 1rfolds3_sl_2/model2 /home/ioannis/evolution/data/1rfolds3_sl/fold2/supervised_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save supervised_predictions.txt &





################################################################################################
################################################################################################
################################################################################################
#
# 1rfolds3_sl_filtered directory of data used for this experiments
#
################################################################################################
################################################################################################
################################################################################################





























##################################
##################################
##################################
##################################
##################################
##################################

##### CONCAT 20 FOLDS #######

# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/concat_folds/fold14/multi_training.json /home/ioannis/evolution/data/concat_folds/fold14/validation.json mnli_con_folds/model14 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/concat_folds/fold14/snli_training.json /home/ioannis/evolution/data/concat_folds/fold14/validation.json snli_con_folds/model14 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt

# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold14/training.json /home/ioannis/evolution/data/folds/fold14/validation.json folds/model19 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt


######## 20 FOLDS TRAINING ########
# old data in 0,2,4 folds
##################################

############### experiment with these folds ##############
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold0/training.json /home/ioannis/evolution/data/folds/fold0/validation.json folds/model8 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold2/training.json /home/ioannis/evolution/data/folds/fold2/validation.json folds/model7 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold4/training.json /home/ioannis/evolution/data/folds/fold4/validation.json folds/model9 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt

# python my_evaluate.py folds/model8 /home/ioannis/evolution/data/folds/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt
# python my_evaluate.py folds/model7 /home/ioannis/evolution/data/folds/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt
# python my_evaluate.py folds/model9 /home/ioannis/evolution/data/folds/fold4/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt

########################################################



# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold0/training.json /home/ioannis/evolution/data/folds/fold0/validation.json folds/model0 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold1/training.json /home/ioannis/evolution/data/folds/fold1/validation.json folds/model1 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold2/training.json /home/ioannis/evolution/data/folds/fold2/validation.json folds/model2 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt

# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold3/training.json /home/ioannis/evolution/data/folds/fold3/validation.json folds/model3 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold4/training.json /home/ioannis/evolution/data/folds/fold4/validation.json folds/model4 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold5/training.json /home/ioannis/evolution/data/folds/fold5/validation.json folds/model5 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt

# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold6/training.json /home/ioannis/evolution/data/folds/fold6/validation.json folds/model6 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt


# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold14/training.json /home/ioannis/evolution/data/folds/fold14/validation.json folds/model14 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold15/training.json /home/ioannis/evolution/data/folds/fold15/validation.json folds/model15 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold16/training.json /home/ioannis/evolution/data/folds/fold16/validation.json folds/model16 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt


######## FOLDS EVALUATION ########
##################################
# python my_evaluate.py binary_snli_2 /home/ioannis/evolution/data/folds/fold0/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt


# python my_evaluate.py folds/model0 /home/ioannis/evolution/data/folds/fold0/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model1 /home/ioannis/evolution/data/folds/fold1/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model2 /home/ioannis/evolution/data/folds/fold2/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt

# python my_evaluate.py folds/model3 /home/ioannis/evolution/data/folds/fold3/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model4 /home/ioannis/evolution/data/folds/fold4/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model5 /home/ioannis/evolution/data/folds/fold5/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt

# python my_evaluate.py folds/model6 /home/ioannis/evolution/data/folds/fold6/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt


# python my_evaluate.py folds/model14 /home/ioannis/evolution/data/folds/fold14/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model15 /home/ioannis/evolution/data/folds/fold15/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model16 /home/ioannis/evolution/data/folds/fold16/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt

######## FOLDS RANKING PROBABILISTIC OUTPUT ########
####################################################
#python interactive-eval.py folds/model0 ~/data/glove/glove-840B.npy -i -a --vocab ~/data/glove/glove-840B-vocabulary.txt

# python my_evaluate.py binary_snli_2 /home/ioannis/evolution/data/folds/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt

# python my_evaluate.py folds/model0 /home/ioannis/evolution/data/folds/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds/model1 /home/ioannis/evolution/data/folds/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds/model2 /home/ioannis/evolution/data/folds/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt

# python my_evaluate.py folds/model3 /home/ioannis/evolution/data/folds/fold3/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds/model4 /home/ioannis/evolution/data/folds/fold4/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds/model5 /home/ioannis/evolution/data/folds/fold5/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt

# python my_evaluate.py folds/model6 /home/ioannis/evolution/data/folds/fold6/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt


# python my_evaluate.py folds/model14 /home/ioannis/evolution/data/folds/fold14/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds/model15 /home/ioannis/evolution/data/folds/fold15/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds/model16 /home/ioannis/evolution/data/folds/fold16/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt


########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################





####### FOLDS 5 ##############


# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds5/fold0/training.json /home/ioannis/evolution/data/folds5/fold0/validation.json folds5/model0 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds5/fold1/training.json /home/ioannis/evolution/data/folds5/fold1/validation.json folds5/model1 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds5/fold2/training.json /home/ioannis/evolution/data/folds5/fold2/validation.json folds5/model2 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds5/fold3/training.json /home/ioannis/evolution/data/folds5/fold3/validation.json folds5/model3 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds5/fold4/training.json /home/ioannis/evolution/data/folds5/fold4/validation.json folds5/model4 mlp --lower -e 30 -u 300 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt



# python my_evaluate.py folds5/model0 /home/ioannis/evolution/data/folds5/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds5/model1 /home/ioannis/evolution/data/folds5/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds5/model2 /home/ioannis/evolution/data/folds5/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py folds5/model3 /home/ioannis/evolution/data/folds5/fold3/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt















########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################





# nohup python -u my_train.py ~/data/glove/glove.840B.300d.txt /home/ioannis/evolution/data/training_pairs.json /home/ioannis/evolution/data/validation_pairs.json my_model mlp --lower -e 30 -u 200 -d 0.8 --l2 0 -b 32 -r 0.05 --optim adagrad &
# nohup python -u my_train.py ~/data/glove/glove.840B.300d.txt /home/ioannis/evolution/data/training_pairs.json /home/ioannis/evolution/data/validation_pairs.json my_model mlp --lower -e 30 -u 200 -d 0.8 --l2 0 -b 32 -r 0.05 --optim adagrad &> nohup2.out&

# python -u my_train.py ~/data/glove/glove.840B.300d.txt /home/ioannis/evolution/data/training_100.json /home/ioannis/evolution/data/validation_100.json my_model mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300




###### this gave 81% accuracy ######
# python -u my_train.py ~/data/glove/glove.840B.300d.txt /home/ioannis/evolution/data/meta_training_110.json /home/ioannis/evolution/data/meta_validation_110.json my_model_1 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300

###### Evaluate on validation data set ######
### training #####
# python my_evaluate.py my_model/ /home/ioannis/evolution/data/meta_training_110.json ~/data/glove/glove.840B.300d.txt
#### validation #####
# python my_evaluate.py /home/ioannis/models/my_model/ /home/ioannis/evolution/data/meta_validation_110.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt



###### this gave 81% accuracy ######
# python -u my_train.py ~/data/glove/glove.840B.300d.txt /home/ioannis/evolution/data/meta_training_111.json /home/ioannis/evolution/data/meta_validation_111.json my_model_111 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300

## Evaluate ##
# python my_evaluate.py my_model_111/ /home/ioannis/evolution/data/meta_validation_111.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_1.txt

## Ranking Evaluation
# python my_evaluate.py my_model_111/ /home/ioannis/evolution/data/meta_ranking_validation_111.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt


###### MY_MODEL_15 ######

# ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/meta_training_115.json /home/ioannis/evolution/data/meta_validation_115.json my_model_115 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/meta_training_115.json /home/ioannis/evolution/data/meta_validation_115.json my_model_115_slow mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt

## Ranking Evaluation
# python my_evaluate.py my_model_115_slow /home/ioannis/evolution/data/meta_ranking_validation_115.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt
# python my_evaluate.py my_model_115 /home/ioannis/evolution/data/meta_ranking_validation_115.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions.txt




###### FAST DEPLOY #######
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/meta_training_110.json /home/ioannis/evolution/data/meta_validation_110.json my_model mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt

####### INFERENCE #######

# python interactive-eval.py ~/models/snli_trained/ ~/data/glove/glove-840B.npy --vocab ~/data/glove/glove-840B-vocabulary.txt -i -a





from __future__ import division, print_function

"""
Script to train an RTE LSTM.
"""

import sys
import argparse
import tensorflow as tf

import ioutils
import my_ioutils
import utils
from classifiers import LSTMClassifier, MultiFeedForwardClassifier,\
    DecomposableNLIModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('embeddings',
                        help='Text or numpy file with word embeddings')
    parser.add_argument('train', help='JSONL or TSV file with training corpus')
    parser.add_argument('validation',
                        help='JSONL or TSV file with validation corpus')
    parser.add_argument('save', help='Directory to save the model files')
    parser.add_argument('model', help='Type of architecture',
                        choices=['lstm', 'mlp'])
    parser.add_argument('--vocab', help='Vocabulary file (only needed if numpy'
                                        'embedding file is given)')
    parser.add_argument('-e', dest='num_epochs', default=10, type=int,
                        help='Number of epochs')
    parser.add_argument('-b', dest='batch_size', default=32, help='Batch size',
                        type=int)
    parser.add_argument('-u', dest='num_units', help='Number of hidden units',
                        default=100, type=int)
    parser.add_argument('--no-proj', help='Do not project input embeddings to '
                                          'the same dimensionality used by '
                                          'internal networks',
                        action='store_false', dest='no_project')
    parser.add_argument('-d', dest='dropout', help='Dropout keep probability',
                        default=1.0, type=float)
    parser.add_argument('-c', dest='clip_norm', help='Norm to clip training '
                                                     'gradients',
                        default=100, type=float)
    parser.add_argument('-r', help='Learning rate', type=float, default=0.001,
                        dest='rate')
    parser.add_argument('--lang', choices=['en', 'pt'], default='en',
                        help='Language (default en; only affects tokenizer)')
    parser.add_argument('--lower', help='Lowercase the corpus (use it if the '
                                        'embedding model is lowercased)',
                        action='store_true')
    parser.add_argument('--use-intra', help='Use intra-sentence attention',
                        action='store_true', dest='use_intra')
    parser.add_argument('--l2', help='L2 normalization constant', type=float,
                        default=0.0)
    parser.add_argument('--report', help='Number of batches between '
                                         'performance reports',
                        default=100, type=int)
    parser.add_argument('-v', help='Verbose', action='store_true',
                        dest='verbose')
    parser.add_argument('--optim', help='Optimizer algorithm',
                        default='adagrad',
                        choices=['adagrad', 'adadelta', 'adam'])

    args = parser.parse_args()

    utils.config_logger(args.verbose)
    logger = utils.get_logger('train')
    logger.debug('Training with following options: %s' % ' '.join(sys.argv))
    train_pairs = my_ioutils.read_corpus(args.train, args.lower, args.lang)
    valid_pairs = my_ioutils.read_corpus(args.validation, args.lower, args.lang)

    # whether to generate embeddings for unknown, padding, null
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings, args.vocab,
                                                    True, normalize=True)

    # print(word_dict)
    logger.info('Converting words to indices')
    # find out which labels are there in the data
    # (more flexible to different datasets)
    label_dict = utils.create_label_dict(train_pairs)
    NUMBER_CLASSES = len(label_dict)
    train_data = utils.create_dataset(train_pairs, word_dict, label_dict)
    valid_data = utils.create_dataset(valid_pairs, word_dict, label_dict)

    ioutils.write_params(args.save, lowercase=args.lower, language=args.lang,
                         model=args.model)
    ioutils.write_label_dict(label_dict, args.save)
    ioutils.write_extra_embeddings(embeddings, args.save)

    msg = '{} sentences have shape {} (firsts) and {} (seconds)'
    logger.debug(msg.format('Training',
                            train_data.sentences1.shape,
                            train_data.sentences2.shape))
    logger.debug(msg.format('Validation',
                            valid_data.sentences1.shape,
                            valid_data.sentences2.shape))

    sess = tf.InteractiveSession()
    logger.info('Creating model')
    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    print("vocab size {0}".format(vocab_size))
    print("embedding_size {0}".format(embedding_size))

    if args.model == 'mlp':
        model = MultiFeedForwardClassifier(args.num_units, NUMBER_CLASSES, vocab_size,
                                           embedding_size,
                                           use_intra_attention=args.use_intra,
                                           training=True,
                                           project_input=args.no_project,
                                           optimizer=args.optim)
    else:
        model = LSTMClassifier(args.num_units, NUMBER_CLASSES, vocab_size,
                               embedding_size, training=True,
                               project_input=args.no_project,
                               optimizer=args.optim)

    model.initialize(sess, embeddings)

    # this assertion is just for type hinting for the IDE
    assert isinstance(model, DecomposableNLIModel)

    total_params = utils.count_parameters()
    logger.debug('Total parameters: %d' % total_params)

    logger.info('Starting training')
    model.train(sess, train_data, valid_data, args.save, args.rate,
                args.num_epochs, args.batch_size, args.dropout, args.l2,
                args.clip_norm, args.report)
