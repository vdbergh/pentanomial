from __future__ import division
import math,sys

def MLE(pdf,s):
    """
This function computes the maximum likelood estimate for
a discrete distribution with expectation value s,
given an observed (i.e. empirical) distribution pdf.

pdf is a list of tuples (ai,pi), i=1,...,N. It is assumed that 
that the ai are strictly ascending, a1<s<aN and p1>0, pN>0.

The theory behind this function can be found in the online 
document

http://hardy.uhasselt.be/Toga/computeLLR.pdf

(see Proposition 1.1).

To solve the equation (1.3) in loc. cit. we use a specialized
Newton solver.
"""
    epsilon=1e-9
    v,w=pdf[0][0],pdf[-1][0]
    l,u=-1/(w-s),1/(s-v)
    f=lambda x:sum([p*(a-s)/(1+x*(a-s)) for a,p in pdf])
    fp=lambda x:sum([-p*(a-s)**2/(1+x*(a-s))**2 for a,p in pdf])
    x=0
    fx=f(x)
    while True:
#        print "s=%3f x=%.10f f(x)=%g [ %3f %3f]" % (s,x,f(x),l,u)
        if fx==0:
            break
        xpre,fxpre=x,fx
        x=x-fx/fp(x)
        if x<=l:
            x=(10*l+xpre)/11
        elif x>=u:
            x=(10*u+xpre)/11
        fx=f(x)
        if abs(x-xpre) <epsilon:
            break
    return [(a,p/(1+x*(a-s))) for a,p in pdf]

def LL(pdf1,pdf2):
    return sum([pdf1[i][1]*math.log(pdf2[i][1]) for i in range(0,len(pdf1))])

def LLR(pdf,s0,s1):
    """
This function computes the generalized log likelihood ratio (divided by N) 
for s=s1 versus s=s0 where pdf is an empirical distribution and 
s is the expectation value of the true distribution.
pdf is a list of pairs (value,probability). 
"""
    return LL(pdf,MLE(pdf,s1))-LL(pdf,MLE(pdf,s0))

def LLR_alt(pdf,s0,s1):
    """
This function computes the approximate generalized log likelihood ratio (divided by N) 
for s=s1 versus s=s0 where pdf is an empirical distribution and 
s is the expectation value of the true distribution.
pdf is a list of pairs (value,probability). See

http://hardy.uhasselt.be/Toga/computeLLR.pdf
"""
    r0,r1=[sum([prob*(value-s)**2 for value,prob in pdf]) for s in (s0,s1)]
    return 1/2*math.log(r0/r1)

def L_(x):
    return 1/(1+10**(-x/400))

def priorize(l):
    """ 
If necessary mix in a small prior for regularization.
"""
    epsilon=1e-3
    N=sum(l)
    s=len(l)
    if sum([k==0 for k in l]):
        return [k*(1-epsilon/N)+epsilon/s for k in l]
    else:
        return l

def LLR_logistic(elo0,elo1,results):
    """ 
This function computes the generalized log-likelihood ratio for "results" 
which should be a list of either length 3 or 5. If the length
is 3 then it should contain the frequencies of L,D,W. If the length
is 5 then it should contain the frequencies of the game pairs
LL,LD+DL,LW+DD+WL,DW+WD,WW.
elo0,elo1 are in logistic elo.
"""
    results=priorize(results)
    s0,s1=[L_(elo) for elo in (elo0,elo1)]
    N=sum(results)
    l=len(results)
    pdf=[(i/(l-1),results[i]/N) for i in range(0,l)]
    return N*LLR(pdf,s0,s1)

