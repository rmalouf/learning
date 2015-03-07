import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

import alias

def explode(cues):
    if isinstance(cues, basestring):
        cues = cues.split('_')
    return {}.fromkeys(cues,True)


def orthoCoding(strs,grams=2,sep=None):

    if not np.iterable(grams):
        grams = [grams]

    result = [ ]

    for str in strs:
    
        cues = [ ]
        str = list(str) 

        for n in grams:
            if n > 1:
                seq = ['#'] + str + ['#']
            else:
                seq = str
            count = max(0, len(seq) - n + 1) 
            cues.extend(''.join(seq[i:i+n]) for i in xrange(count))
        if sep:
            result.append(sep.join(cues))
        else:
            result.append(tuple(cues))

    return result


def danks(data):

    ## Rescorla-Wagner equilibirum (Danks 2003)

    feats = DictVectorizer(dtype=int,sparse=False)

    marginals = data.groupby('Cues',as_index=False).Frequency.sum()
    marginals = marginals.rename(columns={'Frequency':'Total'})

    data = pd.merge(data,marginals,on='Cues')
    
    result = pd.DataFrame()

    for outcome in data.Outcomes.unique():

        yes = data[data.Outcomes==outcome]

        M = feats.fit_transform([explode(c) for c in yes.Cues])
        P = np.diag(yes.Total/sum(yes.Total))
        MTP = M.T.dot(P)
        O = yes.Frequency / yes.Total

        left = MTP.dot(M)
        right = MTP.dot(O)

        V = np.linalg.solve(left,right)

        result[outcome] = V


    result.index = feats.get_feature_names()

    return result

def ndl(data):

    ## Naive discriminative learning (Baayen et al. 2011)

    vec = DictVectorizer(dtype=float,sparse=False)
    D = vec.fit_transform([explode(c) for c in data.Cues]) * data.Frequency[:,np.newaxis]

    # Make co-occurrence matrix C

    n = len(vec.get_feature_names())
    C = np.zeros((n,n))
    for row in D:
        for nz in np.nonzero(row):
            C[nz] += row    

    # Normalize
    
    Z = C.sum(axis=1)
    C1 = C / Z[:,np.newaxis]

    # Make outcome matrix O
    
    out = DictVectorizer(dtype=float,sparse=False)
    X = out.fit_transform([explode(c) for c in data.Outcomes]) * data.Frequency[:,np.newaxis]

    O = np.zeros((len(vec.get_feature_names()),len(out.get_feature_names())))
    for i in xrange(len(X)):
        for nz in np.nonzero(D[i]):
            O[nz] += X[i]

    # Normalize
    
    O1 = O / Z[:,np.newaxis]
    
    # Solve
    
    # (fails if C is singular)    
    # W = np.linalg.solve(C1,O1)

    W = np.linalg.pinv(C1).dot(O1)
    
    return pd.DataFrame(W,columns=out.get_feature_names(),index=vec.get_feature_names())
    
def activation(cues, W):

    A = np.zeros(len(W.columns))
    if isinstance(cues, basestring):
        cues = cues.split('_')
    for cue in cues:
        A += W.loc[cue]
        
    return pd.Series(A,index=W.columns)
    
    
    
def activation(cues, W):

    if isinstance(cues, basestring):
        cues = cues.split('_')
    return W[[(c in cues) for c in W.index]].sum()



def _rwUpdate(W, D, O, Alpha, Beta, Lambda):
    Vtotal = np.dot(W.T, D)
    L = O * Lambda
    Vdelta = Alpha * Beta * (L - Vtotal)
    W += D[:,np.newaxis] * Vdelta

try:
    import _ndl
    rwUpdate = _ndl.rwUpdate
except ImportError:
    rwUpdate = _rwUpdate
        

def rw(data, Alpha=0.1, Beta=0.1, Lambda=1.0, M=50000, distribution=None, trajectory=False):

    # code cues

    cues = DictVectorizer(dtype=np.int, sparse=False)
    D = cues.fit_transform([explode(c) for c in data.Cues])
    
    # code outcomes

    out = DictVectorizer(dtype=np.int, sparse=False)
    O = out.fit_transform([explode(c) for c in data.Outcomes])

    # weight matrix

    W = np.zeros((len(cues.get_feature_names()), len(out.get_feature_names())))

    if distribution is None:
        E = data.Frequency / sum(data.Frequency)
        rand = alias.multinomial(E)
        
    history = dict()

    i = 0
    
    while i < M:
        i += 1
        
        if distribution is None:
            item = rand.draw()
            
        else:
            item = distribution() - 1
            
            while item >= len(data):
                item = distribution() - 1
                
        rwUpdate(W, D[item,:], O[item,:], Alpha, Beta, Lambda)
        
        if trajectory:
            history[i] = pd.DataFrame(W, columns=out.get_feature_names(), index=cues.get_feature_names(), copy=True)

    if trajectory:
        return pd.Panel.from_dict(history)
    else:
        return pd.DataFrame(W, columns=out.get_feature_names(), index=cues.get_feature_names())                
        
        