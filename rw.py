#!/usr/bin/env python

## Rescorla-Wagner learning model

import pandas as pd
import numpy as np
import ndl

def main():

    data = pd.read_csv('lexample.csv')
    W0 = ndl.ndl(data)
    diff = np.zeros_like(W0)
    for i in xrange(10):
        W = ndl.rw(data,M=10000)
        diff += abs(W - W0)
    diff = diff / 10.
    print diff.max(),diff.min(),np.mean(diff)

if __name__ == '__main__':
    main()
                           