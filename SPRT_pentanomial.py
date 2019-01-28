from __future__ import division
import math
import LLRcalc

class SPRT:
    """
    This class performs a GSPRT for H0:elo=elo0 versus H1:elo=elo1
    See here for a description of the GSPRT as well as theoretical
     (asymptotic) results.

    http://stat.columbia.edu/~jcliu/paper/GSPRT_SQA3.pdf

    In addition we do overshoot correction as in Siegmund - Sequential Analysis.

    To record the outcome of a game pair use the method record(result)  
    where "result" is a half integer in the interval [0,2]
"""

    def __init__(self,alpha=0.05,beta=0.05,elo0=0,elo1=5,mode='pentanomial'):
        self.elo0=elo0
        self.elo1=elo1
        assert(mode in ('trinomial','pentanomial'))
        if mode=='pentanomial':
            self.results_=5*[0]
        else:
            self.results_=3*[0]
        self.LA=math.log(beta/(1-alpha))
        self.LB=math.log((1-beta)/alpha)
        self.status_=''
        self.LLR_=0

    def record(self,result):
        if self.status_!='':
            return
        self.results_[result]+=1
        self.LLR_,overshoot=LLRcalc.LLR_logistic(self.elo0,self.elo1,self.results_)
        # sanitize
        overshoot=min((self.LB-self.LA)/20,overshoot)
        if self.LLR_>self.LB-overshoot:
            self.status_='H1'
        elif self.LLR_ < self.LA+overshoot:
            self.status_='H0'

    def status(self):
        return self.status_

    def length(self):
        l=len(self.results_)
        return ((l-1)/2)*sum(self.results_)

    def LLR(self):
        return self.LLR_

    def results(self):
        return self.results_
