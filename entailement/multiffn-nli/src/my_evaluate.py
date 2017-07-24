# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
from itertools import izip
import tensorflow as tf
import my_ioutils
import utils
import ioutils
import IPython

"""
Evaluate the performance of an NLI model on a dataset
"""


def print_errors(pairs, answers, label_dict, probabilities):
    """
    Print the pairs for which the model gave a wrong answer,
    their gold label and the system one.
    """
    for pair, answer, prob in izip(pairs, answers, probabilities):
        label_str = pair[2]
        label_number = label_dict[label_str]
        # if answer != label_number:
        sent1 = ' '.join(pair[0])
        sent2 = ' '.join(pair[1])
        print('Sent 1: {}\nSent 2: {}'.format(sent1, sent2))
        print('System label: {}, True label: {} - {} ##Pro {}##'.format(answer,
                                                            label_number, label_str,prob))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory with saved model')
    parser.add_argument('dataset',
                        help='JSONL or TSV file with data to evaluate on')
    parser.add_argument('embeddings', help='Numpy embeddings file')
    parser.add_argument('--vocabulary',
                        help='Text file with embeddings vocabulary')
    parser.add_argument('-v',
                        help='Verbose', action='store_true', dest='verbose')
    parser.add_argument('-e',
                        help='Print pairs and labels that got a wrong answer',
                        action='store_true', dest='errors')
    args = parser.parse_args()

    utils.config_logger(verbose=args.verbose)
    params = ioutils.load_params(args.model)
    sess = tf.InteractiveSession()

    model_class = utils.get_model_class(params)
    model = model_class.load(args.model, sess)
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings,
                                                    args.vocabulary,
                                                    generate=False,
                                                    load_extra_from=args.model,
                                                    normalize=True)
    model.initialize_embeddings(sess, embeddings)
    label_dict = ioutils.load_label_dict(args.model)

    pairs = my_ioutils.read_corpus(args.dataset, params['lowercase'],
                                params['language'])
    dataset = utils.create_dataset(pairs, word_dict, label_dict)
    loss, acc, answers, probabilities = model.evaluate(sess, dataset, True, 32, testing_mode=True)
    # IPython.embed()
    formated_probabilities = [prob_tuple for batch in probabilities for prob_tuple in batch ]
    if args.errors:
        print_errors(pairs, answers, label_dict, formated_probabilities)
    print(label_dict)
    print(len(formated_probabilities))
    for i, a in enumerate(answers):
        print a, formated_probabilities[i][label_dict['entailement']]
    print('Loss: %f' % loss)
    print('Accuracy: %f' % acc)

