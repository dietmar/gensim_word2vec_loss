import argparse
import logging
import multiprocessing
import os
import re
import sqlite3
import subprocess
import sys

import numpy
from gensim.models import word2vec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DLURL = 'https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/million_post_corpus.tar.bz2'
DB = 'million_post_corpus/corpus.sqlite3'
MAXCOUNT = 100000
EPOCHS = 10
MAXALPHA = 0.1
MINALPHA = 0.001
EMB_SIZE = 300

def micro_tokenize(txt):
    words = []
    # split at whitespace
    for w in txt.split():
        w = w.strip('.,!?:;"-+()„“”»«…\'`’*')
        # words need to contain at least one "regular" character
        if re.search(r'[a-zöüßA-ZÄÖÜ]', w):
            words.append(w)
    return words

def normalize(txt):
    txt = txt.lower()

    # replace URLs
    url_re1 = re.compile(r'(?:ftp|http)s?://[\w\d:#@%/;$()~_?+=\,.&#!|-]+')
    txt = url_re1.sub('URL', txt)
    url_re2 = re.compile(r'\bwww\.[a-zA-Z0-9-]{2,63}\.[\w\d:#@%/;$()~_?+=\,.&#!|-]+')
    txt = url_re2.sub('URL', txt)
    url_re3 = re.compile(r'\b[a-zA-Z0-9.]+\.(?:com|org|net|io)')
    txt = url_re3.sub('URL', txt)

    # replace emoticons
    # "Western" emoticons such as =-D and (^:
    # inspired by http://sentiment.christopherpotts.net/tokenizing.html
    s = r"(^|\s)"                # beginning or whitespace required before
    s += r"(?:"                  # begin emoticon
    s += r"(?:"                  # begin "forward" emoticons like :-)
    s += r"[<>]?"                # optinal hat/brow
    s += r"[:;=8xX]"             # eyes
    s += r"[o*'^-]?"             # optional nose
    s += r"[(){}[\]dDpP/\\|@3]+" # mouth
    s += r")"                    # end "forward" emoticons
    s += r"|"                    # or
    s += r"(?:"                  # begin "backward" emoticons like (-:
    s += r"[(){}[\]dDpP/\\|@3]+" # mouth
    s += r"[o*'^-]?"             # optional nose
    s += r"[:;=8xX]"             # eyes
    s += r"[<>]?"                # optinal hat/brow
    s += r")"                    # end "backward" emoticons
    # "Eastern" emoticons like ^^ and o_O
    s += r"|"                    # or
    s += "(?:\^\^)|(?:o_O))"     # only two eastern emoticons for now
    s += r"(\s|$)"               # white space or end required after
    emoticon_re = re.compile(s)
    txt = emoticon_re.sub(r'\1EMOTICON\2', txt)

    # remove repeated symbols
    for s in ',.!?:;#-_=+*/$@%<>&()[]':
        txt = re.sub('[%s]+' % s, s, txt)

    # separate punctuation
    txt = re.sub(r'([.,!?:;/()\'"„“”»«`’…$%*])', r' \1 ', txt)

    # remove leading, trailing and repeated whitespace
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)

    return txt

def data_generator(maxcount):
    con = sqlite3.connect(DB)
    sql = '''
        SELECT COALESCE(Headline, '') || ' ' || COALESCE(Body, '')
        FROM Posts
        ORDER BY ID_Post
    '''
    r = con.execute(sql)
    pool = multiprocessing.Pool()
    count = 0
    while True:
        rows = r.fetchmany(1000)
        if len(rows) == 0:
            break
        wordlists = pool.map(micro_tokenize,
            pool.map(normalize, [ row[0] for row in rows ]))
        for wl in wordlists:
            yield wl
            count += 1
            if count >= maxcount:
                return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxalpha', type=float, default=MAXALPHA)
    parser.add_argument('--minalpha', type=float, default=MINALPHA)
    args = parser.parse_args()

    if not os.path.exists(DB):
        print('Downloading data (will use 340M disk space) ... ',
            end='', flush=True)
        cmnd = 'wget -q -O- "%s" | tar -xj' % DLURL
        subprocess.check_call(cmnd, shell=True)
        print('done.', flush=True)

    logging.basicConfig(format='%(asctime)s [doc2vec] : %(message)s',
        level=logging.INFO)

    alphas = numpy.linspace(args.maxalpha, MINALPHA, EPOCHS)
    model = word2vec.Word2Vec(
        min_count=5,
        sg=0,
        hs=1,
        iter=1,
        size=EMB_SIZE,
        alpha=alphas[0],
        min_alpha=alphas[-1],
        workers=multiprocessing.cpu_count(),
    )
    model.build_vocab(data_generator(MAXCOUNT))
    losses = []
    for i in range(EPOCHS):
        model.train(
            data_generator(MAXCOUNT),
            compute_loss=True,
            total_examples=model.corpus_count,
            epochs=model.iter,
            start_alpha=alphas[i],
            end_alpha=alphas[i],
        )
        training_loss_val = model.get_latest_training_loss()
        print('After epoch %d: latest training loss is %f' %
            (i + 1, training_loss_val))
        losses.append(training_loss_val)

    f, ax = plt.subplots(1, 1)
    ax.plot(range(1, EPOCHS + 1), losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    f.tight_layout()
    f.savefig('word2vec_loss.png')
