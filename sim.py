#!/usr/bin/env python

## R-W learning simulations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ndl


#from IPython.parallel import Client

#rc = Client(profile='home')
#dview = rc.direct_view()
#dview.block = True
#lview = rc.load_balanced_view()
#lview.block = True
#rc.ids

class Simulation(object):

    def __init__(self, func, data, MAX_M=250, P=100):
    
        self.func = func
        self.MAX_M = MAX_M
        self.P = P
        self.data = data
        
    def activation(self, W):
        return pd.DataFrame([ndl.activation(c, W) for c in self.data.Cues], index=self.data.index)

    def accuracy(self,M):
        W = ndl.rw(self.data, M=M)
        A = self.activation(W)
        return np.mean(A.idxmax(1) == self.data['Outcomes'])

    def population_accuracy(self, M):
        return np.mean([self.accuracy(M) == 1 for i in xrange(self.P)])


def run(args):
    (S,M) = args
    return S.population_accuracy(M)

        
def result(P,exp):
    plt.plot(range(1,len(P[exp])+1), P[exp], '-', linewidth=2)
    plt.title(exp)
    plt.xlabel('Trial Number')
    #plt.yscale('log')
    plt.suptitle('Proportion of learners who label all items correctly')

def experiment(data, M=250, P=100, view=None):
    result = { }
    for func in [ sg_pl, sg_du_pl, sg_du_tr_pl, sg_du_tr_qu_pl, du_notdu ]:
        tmp = pd.DataFrame(data)
        tmp['Outcomes'] = [func(i) for i in tmp['Number']]
        project = Simulation(func, tmp, MAX_M=M, P=P)
        if view:
            result[func.__name__] = view.map(run, ((project, i) for i in xrange(1,M)))
        else:
            result[func.__name__] = map(run, ((project, i) for i in xrange(1,M)))
    return result
    
def all_results(r):
    for exp in r.keys():
        plt.plot(range(1,len(r[exp])+1), r[exp], '-', linewidth=1.5, label=exp)
    
    plt.suptitle('Proportion of learners who label all items correctly')
    plt.xlabel('Trials')
    plt.legend(loc=(-0.55,0.5))    
    
    
## Number marking systems

def sg_pl(i):
    if i == 1:
        return 'sg'
    else:
        return 'pl'

def sg_du_pl(i):
    if i == 1:
        return 'sg'
    elif i == 2:
        return 'du'
    else:
        return 'pl'

def sg_du_tr_pl(i):
    if i == 1:
        return 'sg'
    elif i == 2:
        return 'du'
    elif i == 3:
        return 'tr'
    else:
        return 'pl'

def sg_du_tr_qu_pl(i):
    if i == 1:
        return 'sg'
    elif i == 2:
        return 'du'
    elif i == 3:
        return 'tr'
    elif i == 4:
        return 'qu'
    else:
        return 'pl'

def du_notdu(i):
    if i == 2:
        return 'du'
    else:
        return 'notdu'
