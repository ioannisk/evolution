#!/usr/bin/python
import sys
import os
import numpy
import cPickle as pkl
import json

from collections import OrderedDict

dic = {'entailment': '0', 'contradiction': '1'}

def build_dictionary(filepaths, dst_path, lowercase=False):
    word_freqs = OrderedDict()
    for filepath in filepaths:
        print 'Processing', filepath
        with open(filepath, 'r') as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    words = word_freqs.keys()
    freqs = word_freqs.values()

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['_PAD_'] = 0 # default, padding
    worddict['_UNK_'] = 1 # out-of-vocabulary
    worddict['_BOS_'] = 2 # begin of sentence token
    worddict['_EOS_'] = 3 # end of sentence token

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + 4

    with open(dst_path, 'wb') as f:
        pkl.dump(worddict, f)

    print 'Dict size', len(worddict)
    print 'Done'


def build_sequence(filepath, dst_dir):
    filename = os.path.basename(filepath)
    print filename
    len_p = []
    len_h = []
    with open(filepath) as f, \
         open(os.path.join(dst_dir, 'premise_%s'%filename), 'w') as f1, \
         open(os.path.join(dst_dir, 'hypothesis_%s'%filename), 'w') as f2,  \
         open(os.path.join(dst_dir, 'label_%s'%filename), 'w') as f3:
        # next(f) # skip the header row
        for line in f:
            line = json.loads(line.strip())
            label_= line['class']
            # sents = line.strip().split('\t')
            # words_in = sents[1].strip().split(' ')
            # words_in = [x for x in words_in if x not in ('(',')')]
            des_ = line['des']
            f1.write(des + '\n')
            len_p.append(len(des.split()))

            # words_in = sents[2].strip().split(' ')
            # words_in = [x for x in words_in if x not in ('(',')')]
            web_ = line['web']
            f2.write(web + '\n')
            len_h.append(web.split())

            f3.write(dic[label_] + '\n')

    print 'max min len premise', max(len_p), min(len_p)
    print 'max min len hypothesis', max(len_h), min(len_h)


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing fold{} dataset'.format(FOLD)))
    print('=' * 80)
    FOLD = 0
    base_dir = "/home/ioannis/data/recovery_test/fold{}".format(FOLD)
    dst_dir = os.path.join(base_dir, 'web_word_sequence')
    make_dirs([dst_dir])

    build_sequence(os.path.join(base_dir, 'training.json'), dst_dir)
    build_sequence(os.path.join(base_dir, 'validation.json'), dst_dir)
    build_sequence(os.path.join(base_dir, 'ranking_validation.json'), dst_dir)

    build_dictionary([os.path.join(dst_dir, 'premise_snli_1.0_train.txt'),
                      os.path.join(dst_dir, 'hypothesis_snli_1.0_train.txt')],
                      os.path.join(dst_dir, 'vocab_cased.pkl'))

