from __future__ import division
import math,random
import LLRcalc

def L(x):
    return 1/(1+10**(-x/400))

class SPRT:
    """
    This class performs a GSPRT for H0:elo=elo0 versus H1:elo=elo1
    See here for a description of the GSPRT as well as theoretical
     (asymptotic) results.

    http://stat.columbia.edu/~jcliu/paper/GSPRT_SQA3.pdf

    To record the outcome of a game pair use the method record(result)  
    where "result" is a half integer in the interval [0,2]
"""

    def __init__(self,alpha=0.05,beta=0.05,elo0=0,elo1=5,mode='pentanomial'):
        self.alpha=alpha
        self.beta=beta
        self.elo0=elo0
        self.elo1=elo1
        assert(mode in ('trinomial','pentanomial'))
        if mode=='pentanomial':
            self.results=5*[0]
        else:
            self.results=3*[0]
        self.status_=''
        self.LLR_=0

    def record(self,result):
        if self.status_!='':
            return
        self.results[result]+=1
        self.LLR_,prob,status=LLRcalc.sprt(self.alpha,self.beta,self.elo0,self.elo1,self.results)
        if status=='':
            return
        p=random.random()
        if p>prob:
            return
#        print(self.LLR_,prob,status)
        self.status_=status

    def status(self):
        return self.status_

    def length(self):
        l=len(self.results)
        return ((l-1)/2)*sum(self.results)

    def LLR(self):
        return self.LLR_

