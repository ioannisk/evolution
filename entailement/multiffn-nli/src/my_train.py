# -*- coding: utf-8 -*-

#20 fold stats
#(1902, 14), (1931, 15), (1959, 16), (1986, 17), (2015, 18), (2052, 19)


######## FOLDS TRAINING ########
##################################
## not enough training

# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold0/training.json /home/ioannis/evolution/data/folds/fold0/validation.json folds/model0 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold2/training.json /home/ioannis/evolution/data/folds/fold2/validation.json folds/model2 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold4/training.json /home/ioannis/evolution/data/folds/fold4/validation.json folds/model4 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt






# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold0/training.json /home/ioannis/evolution/data/folds/fold0/validation.json folds/model0 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold1/training.json /home/ioannis/evolution/data/folds/fold1/validation.json folds/model1 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold2/training.json /home/ioannis/evolution/data/folds/fold2/validation.json folds/model2 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt

# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold3/training.json /home/ioannis/evolution/data/folds/fold3/validation.json folds/model3 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold4/training.json /home/ioannis/evolution/data/folds/fold4/validation.json folds/model4 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold5/training.json /home/ioannis/evolution/data/folds/fold5/validation.json folds/model5 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt


# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold14/training.json /home/ioannis/evolution/data/folds/fold14/validation.json folds/model14 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold15/training.json /home/ioannis/evolution/data/folds/fold15/validation.json folds/model15 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/folds/fold16/training.json /home/ioannis/evolution/data/folds/fold16/validation.json folds/model16 mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.005 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt


######## FOLDS EVALUATION ########
##################################
# python my_evaluate.py folds/model0 /home/ioannis/evolution/data/folds/fold0/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model1 /home/ioannis/evolution/data/folds/fold1/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model2 /home/ioannis/evolution/data/folds/fold2/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt

# python my_evaluate.py folds/model3 /home/ioannis/evolution/data/folds/fold3/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model4 /home/ioannis/evolution/data/folds/fold4/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model5 /home/ioannis/evolution/data/folds/fold5/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt


# python my_evaluate.py folds/model14 /home/ioannis/evolution/data/folds/fold14/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model15 /home/ioannis/evolution/data/folds/fold15/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt
# python my_evaluate.py folds/model16 /home/ioannis/evolution/data/folds/fold16/validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt

######## FOLDS RANKING PROBABILISTIC OUTPUT ########
####################################################
# python my_evaluate.py folds/model0 /home/ioannis/evolution/data/folds/fold0/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt
# python my_evaluate.py folds/model1 /home/ioannis/evolution/data/folds/fold1/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt
# python my_evaluate.py folds/model2 /home/ioannis/evolution/data/folds/fold2/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt

# python my_evaluate.py folds/model3 /home/ioannis/evolution/data/folds/fold3/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt
# python my_evaluate.py folds/model4 /home/ioannis/evolution/data/folds/fold4/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt
# python my_evaluate.py folds/model5 /home/ioannis/evolution/data/folds/fold5/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt


# python my_evaluate.py folds/model14 /home/ioannis/evolution/data/folds/fold14/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt
# python my_evaluate.py folds/model15 /home/ioannis/evolution/data/folds/fold15/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt
# python my_evaluate.py folds/model16 /home/ioannis/evolution/data/folds/fold16/ranking_validation.json ~/data/glove/glove-840B.npy --vocabulary ~/data/glove/glove-840B-vocabulary.txt -save prob_predictions_lr.txt


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




###### FAST DEPLOY #######
# python -u my_train.py ~/data/glove/glove-840B.npy /home/ioannis/evolution/data/meta_training_110.json /home/ioannis/evolution/data/meta_validation_110.json my_model mlp --lower -e 30 -u 200 -d 0.8  -b 32 -r 0.05 --report 300 --vocab ~/data/glove/glove-840B-vocabulary.txt

####### INFERENCE #######
# python interactive-eval.py my_model/ ~/data/glove/glove.840B.300d.txt -i -a
# python interactive-eval.py my_model/ ~/data/glove/fast_glove.txt -i -a




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
