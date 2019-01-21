from __future__ import division

import math,sys,argparse

import stats_pentanomial

"""
This program computes passing probabilities and expected running times for SPRT tests.

Based on:

[W1] A. Wald, Sequential analysis.
"""
bb=math.log(10)/400

def L(x):
    return 1/(1+math.exp(-bb*x))

class SPRT:

    def __init__(self,alpha=0.05,beta=0.05,elo0=0,elo1=5,var=None):
        self.score0=L(elo0)
        self.score1=L(elo1)
        self.alpha=alpha
        self.beta=beta
        self.var=var
        self.sigma2=(self.score1-self.score0)**2/self.var

    def mu(self,elo_diff):
        s=L(elo_diff)
        s0=self.score0
        s1=self.score1
        return (s1-s0)*(s-(s0+s1)/2)/self.var

    def characteristics(self,elo_diff):
        """ 
Expected running time and power of SPRT test using Brownian approximation.
See e.g. [W1].
"""
        mu=self.mu(elo_diff)
        sigma2=self.sigma2
        coeff=2*mu/sigma2
        alpha=self.alpha
        beta=self.beta
        LA=math.log(beta/(1-alpha))
        LB=math.log((1-beta)/alpha)
        exp_a=math.exp(-coeff*LA)
        exp_b=math.exp(-coeff*LB)
# avoid division by zero
        if abs(coeff*(LA-LB))<1e-6:
            E=-LA*LB/sigma2
            prob_H1=(0-LA)/(LB-LA)
        else:
            E=-(mu**(-1))*(-LA*exp_b+LB*exp_a-(LB-LA))/(exp_b-exp_a)
            prob_H1=(1-exp_a)/(exp_b-exp_a)

        return (prob_H1,E)
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--alpha",help="probability of a false positve",type=float,default=0.05)
    parser.add_argument("--beta" ,help="probability of a false negative",type=float,default=0.05)
    parser.add_argument("--elo0", help="H0 (expressed in logistic elo)",type=float,default=0.0)
    parser.add_argument("--elo1", help="H1 (expressed in logistic elo)",type=float,default=5.0)
    parser.add_argument("--elo", help="actual logistic elo",type=float,required=True)
    parser.add_argument("--var", help="variance",type=float,default=0.0825)
    args=parser.parse_args()
    alpha=args.alpha
    beta=args.beta
    elo0=args.elo0
    elo1=args.elo1
    elo=args.elo
    var=args.var

    print("elo0=   %.2f" % elo0)
    print("elo1=   %.2f" % elo1)
    print("elo=    %.2f" % elo)
    print("var=    %.4f" % var)

    alpha=0.05
    beta=0.05

    s=SPRT(alpha,beta,elo0,elo1,var)
    (power,expected)=s.characteristics(elo)

    print("pass probability:      %4.2f%%" % (100*power))
    print("avg running time: %10.0f" % expected)

    
