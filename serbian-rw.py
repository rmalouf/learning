#!/usr/bin/env python

## Rescorla-Wagner learning model

from itertools import repeat
from time import time

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

import ndl

from multiprocessing import Pool

def predict(data,weights):

    num = ['Sg','Pl']
    case = ['nom','gen','dat','acc','ins','loc']
    infl = num + case
    predict = [ ]
    for cue in data.Cues:
        A = ndl.activation(cue,weights)
        A.sort(ascending=False)
        res = [ None, None, None ]
        for ind in A.index:
            if ind in num:
                res[2] = ind
            elif ind in case:
                res[1] = ind
            else:
                res[0] = ind
            if not None in res:
                break
        predict.append(tuple(res))
    return predict

def simulate(i):

    global data
    W = ndl.rw(data,M=100000)
    return i,W

def main():

    global data
    
    # baseline using equilibrium equations
    data = pd.read_csv('serbian.csv')
    W0 = ndl.ndl(data)
    diff = np.zeros_like(W0)
    
    R = 1000

    P = Pool(3)
    for i,W in P.imap_unordered(simulate,xrange(R)):
        diff += abs(W - W0)
        print i,diff.max(),diff.min(),np.mean(diff)
    diff = diff / R
    print diff.max(),diff.min(),np.mean(diff)

if __name__ == '__main__':
    main()

