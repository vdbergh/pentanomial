from __future__ import division
from scipy.optimize import brentq
import math,sys,copy

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

"""
    epsilon=1e-9
    v,w=pdf[0][0],pdf[-1][0]
    l,u=-1/(w-s),1/(s-v)
    f=lambda x:sum([p*(a-s)/(1+x*(a-s)) for a,p in pdf])
    x,res=brentq(f,l+epsilon,u-epsilon,full_output=True)
    assert(res.converged)
    pdf_MLE=[(a,p/(1+x*(a-s))) for a,p in pdf]
    s_,var=stats(pdf_MLE) # for validation
    assert(abs(s-s_)<1e-6)
    return pdf_MLE

def stats(pdf):
    epsilon=1e-6
    for i in range(0,len(pdf)):
        assert(-epsilon<=pdf[i][1]<=1+epsilon)
    n=sum([prob for value,prob in pdf])
    assert(abs(n-1)<epsilon)
    s=sum([prob*value for value,prob in pdf])
    var=sum([prob*(value-s)**2 for value,prob in pdf])
    return s,var

def LLjumps(pdf,s0,s1):
    pdf0,pdf1=[MLE(pdf,s) for s in (s0,s1)]
    return [(math.log(pdf1[i][1])-math.log(pdf0[i][1]),pdf[i][1]) for i in range(0,len(pdf))]
        
def LLR(pdf,s0,s1):
    """
This function computes the generalized log likelihood ratio (divided by N)
for s=s1 versus s=s0 where pdf is an empirical distribution and
s is the expectation value of the true distribution.
pdf is a list of pairs (value,probability). 
"""
    return stats(LLjumps(pdf,s0,s1))[0]

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

def LLR_alt2(pdf,s0,s1):
    """
This function computes the approximate generalized log likelihood ratio (divided by N)
for s=s1 versus s=s0 where pdf is an empirical distribution and
s is the expectation value of the true distribution.
pdf is a list of pairs (value,probability). See

http://hardy.uhasselt.be/Toga/GSPRT_approximation.pdf
"""
    s,var=stats(pdf)
    return (s1-s0)*(2*s-s0-s1)/var/2.0

def stopping_rule(N,pdf,A,B,jumps):
    """
This implements a randomized stopping procedure for
a random walk with boundaries A,B such that on average
the random walk will stop (almost) exactly on A or B.

N is the number of observation and pdf is the empirical
distribution of the jumps.

The return value is a triple (x,prob,status) where
x is the current location (which can be outside the
interval [A,B]) and prob is the probablility that the
walk should be stopped at this point (it will be 1
if x is outside the interval, and 0 if in the next
step x cannot overstep the boundary, in the other cases
it will be a value in ]0,1[. Finally status
is the result of the walk if it is stopped at this point.
"""
    jumps=sorted(jumps)
    x=N*stats(jumps)[0]
    v,w=jumps[0][0],jumps[-1][0]
    if x<=A:
        return x,1,'H0'
    if x>=B:
        return x,1,'H1'
    if True:
        if x+v<A:
            D=x-A
            X=-sum([jumps[i][1]*(jumps[i][0]+D) for i in range(0,len(jumps)) if jumps[i][0]+D<0])
            assert(D>=0 and X>=0)
            p=X/(X+D)
            assert(0<=p<=1)
            return x,p,'H0'
        if x+w>B:
            D=B-x
            X=sum([jumps[i][1]*(jumps[i][0]-D) for i in range(0,len(jumps)) if jumps[i][0]-D>0])
            assert(D>=0 and X>=0)
            p=X/(X+D)
            assert(0<=p<=1)
            return x,p,'H1'
    return x,0,''

def L_(x):
    return 1/(1+10**(-x/400))

def regularize(l):
    """ 
If necessary mix in a small prior for regularization.
"""
    epsilon=1e-3
    l=copy.copy(l)
    for i in range(0,len(l)):
        if l[i]==0:
            l[i]=epsilon
    return l

def results_to_pdf(results):
    results=regularize(results)
    N=sum(results)
    l=len(results)
    return N,[(i/(l-1),results[i]/N) for i in range(0,l)]
    
def sprt(alpha,beta,elo0,elo1,results):
    N,pdf=results_to_pdf(results)
    s0,s1=[L_(elo) for elo in (elo0,elo1)]
    jumps=sorted(LLjumps(pdf,s0,s1))
    LA=math.log(beta/(1-alpha))
    LB=math.log((1-beta)/alpha)
    return stopping_rule(N,pdf,LA,LB,jumps)

def LLR_logistic(elo0,elo1,results):
    """ 
This function computes the generalized log-likelihood ratio for "results" 
which should be a list of either length 3 or 5. If the length
is 3 then it should contain the frequencies of L,D,W. If the length
is 5 then it should contain the frequencies of the game pairs
LL,LD+DL,LW+DD+WL,DW+WD,WW.
elo0,elo1 are in logistic elo.
"""
    s0,s1=[L_(elo) for elo in (elo0,elo1)]
    N,pdf=results_to_pdf(results)
    s,var=stats(pdf)
    overshoot=0.583*(s1-s0)/math.sqrt(var)
    return N*LLR(pdf,s0,s1),overshoot

