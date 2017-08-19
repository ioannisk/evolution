import numpy
import os
import sys
from main_evaluate import train
# from main import train


if __name__ == '__main__':
    # model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    model_name = sys.argv[1]
    FOLD = sys.argv[2]

    train(
    saveto           = './{}.npz'.format(model_name),
    reload_          = True,
    dim_word         = 300,
    dim              = 300,
    patience         = 7,
    n_words          = 42394,
    decay_c          = 0.,
    clip_c           = 10.,
    lrate            = 0.0004,
    optimizer        = 'adam',
    maxlen           = 100,
    batch_size       = 32,
    valid_batch_size = 32,
    dispFreq         = 300,
    validFreq        = int(1500),
    saveFreq         = int(1500),
    use_dropout      = True,
    verbose          = False,
    datasets         = ['/home/ioannis/data/recovery_test/fold{}/web_word_sequence/premise_training.json'.format(FOLD),
                        '/home/ioannis/data/recovery_test/fold{}/web_word_sequence/hypothesis_training.json'.format(FOLD),
                        '/home/ioannis/data/recovery_test/fold{}/web_word_sequence/label_training.json'.format(FOLD)],
    valid_datasets   = ['/home/ioannis/data/recovery_test/fold{}/web_word_sequence/premise_validation.json'.format(FOLD),
                        '/home/ioannis/data/recovery_test/fold{}/web_word_sequence/hypothesis_validation.json'.format(FOLD),
                        '/home/ioannis/data/recovery_test/fold{}/web_word_sequence/label_validation.json'.format(FOLD)],
    # test_datasets    = ['../../data/word_sequence/premise_snli_1.0_test.txt',
    #                     '../../data/word_sequence/hypothesis_snli_1.0_test.txt',
    #                     '../../data/word_sequence/label_snli_1.0_test.txt'],
    dictionary       = '../../data/word_sequence/vocab_cased.pkl',
    embedding        = '../../data/glove/glove.840B.300d.txt',
    )

