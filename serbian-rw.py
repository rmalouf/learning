#!/usr/bin/env python

## Rescorla-Wagner learning model

import os,sys
from time import time
from multiprocessing import Pool

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

import ndl

# http://stackoverflow.com/questions/15639779
os.system("taskset -p 0xff %d" % os.getpid())

def simulate(i):
    global data
    W = ndl.rw(data,M=1000000)
    return i,W

def main():

    global data
    
    # baseline using equilibrium equations
    data = pd.read_csv('serbian.csv')
    W0 = ndl.ndl(data)
    diff = np.zeros_like(W0)
    W = np.zeros_like(W0)
    
    # simulate learning for R individuals
    R = 1000
    now = time()
    P = Pool(6)
    for i,W1 in P.imap_unordered(simulate,xrange(R)):
        diff += abs(W1 - W0)
        W += W1
        print >>sys.stderr,i,time()-now
    diff = diff / R
    W = W / R

    # get cue-outcome co-occurrence frequencies
    cues = DictVectorizer(dtype=int,sparse=False)
    D = cues.fit_transform([ndl.explode(c) for c in data.Cues])
    out = DictVectorizer(dtype=int,sparse=False)
    X = out.fit_transform([ndl.explode(c) for c in data.Outcomes]) * data.Frequency[:,np.newaxis]
    O = np.zeros_like(W0)
    for i in xrange(len(X)):
        for nz in np.nonzero(D[i]):
            O[nz] += X[i]

    # save results
    np.savez('serbian-rw',diff=diff,W0=W0.as_matrix(),O=O,W=W)

if __name__ == '__main__':
    main()

