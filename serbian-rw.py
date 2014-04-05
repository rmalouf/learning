#!/usr/bin/env python

## Rescorla-Wagner learning model

from itertools import izip
from time import time

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

import ndl
import rw

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

def main():

    data = pd.read_csv('serbian.csv')

    # baseline using equilibrium equations

    W0 = ndl.ndl(data)
    #P0 = predict(data,W0)

    diff = np.zeros_like(W0)
    acc = 0.0

    for i in xrange(10):
        W = rw.train(data,M=100)
        diff += abs(W - W0)
        #P = predict(data,W)
        #acc += sum(p == p0 for p,p0 in izip(P,P0)) / float(len(P))
        print i,acc
    diff = diff / 10.
    acc = acc / 10.
    print acc,diff.max(),diff.min(),np.mean(diff)

if __name__ == '__main__':
    main()
                           