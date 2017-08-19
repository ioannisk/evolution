#!/usr/bin/python
import sys
import os
import numpy
import cPickle as pkl
import json
import re
from collections import OrderedDict

dic = {'entailment': '0', 'contradiction': '1'}

def clean_up_txt(page_txt):
    page_txt = page_txt.lower()
    page_txt = re.sub('\s+',' ',page_txt)
    # page_txt = re.sub('[^0-9a-zA-Z]+', " ", page_txt)
    page_txt = re.sub('[^a-zA-Z]+', " ", page_txt)
    return page_txt


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
            des_ = clean_up_txt(des_)
            web_ = line['web']
            web_ = clean_up_txt(web_)
            if len(des_.split()) > 125 or len(web_.split()) >125:
                continue

            f1.write(des_ + '\n')
            len_p.append(len(des_.split()))

            # words_in = sents[2].strip().split(' ')
            # words_in = [x for x in words_in if x not in ('(',')')]
            f2.write(web_ + '\n')
            len_h.append(len(web_.split()))

            f3.write(dic[label_] + '\n')

    print 'max min len premise', max(len_p), min(len_p)
    print 'max min len hypothesis', max(len_h), min(len_h)


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

if __name__ == '__main__':
    FOLD = 1
    print('=' * 80)
    print('Preprocessing fold{} dataset'.format(FOLD))
    print('=' * 80)
    base_dir = "/home/ioannis/data/recovery_test/fold{}".format(FOLD)
    dst_dir = os.path.join(base_dir, 'web_word_sequence')
    make_dirs([dst_dir])

    print("trainign.json")
    build_sequence(os.path.join(base_dir, 'training.json'), dst_dir)
    print("validation.json")
    build_sequence(os.path.join(base_dir, 'validation.json'), dst_dir)
    print("ranking_validation.json")
    build_sequence(os.path.join(base_dir, 'ranking_validation.json'), dst_dir)

    build_dictionary([os.path.join(dst_dir, 'premise_training.json'),
                      os.path.join(dst_dir, 'hypothesis_training.json')],
                      os.path.join(dst_dir, 'vocab_cased.pkl'))

