from __future__ import division

import math,sys,argparse

import stats_pentanomial,model_be

"""
This program computes passing probabilities and expected running times for SPRT tests.

Based on:

[W1] A. Wald, Sequential analysis.
"""
bb=math.log(10)/400

def L(x):
    return 1/(1+math.exp(-bb*x))

class SPRT:

    def __init__(self,alpha=0.05,beta=0.05,elo0=0,elo1=5,draw_elo=280,biases=[0],mode='pentanomial'):
        self.score0=L(elo0)
        self.score1=L(elo1)
        self.alpha=alpha
        self.beta=beta
        self.mode=mode
        stats_=model_be.stats_logistic(0,draw_elo,biases)
        assert(mode in ('trinomial','pentanomial'))
        if mode=='trinomial':
            self.ratio=stats_['ratio']
        else:
            self.ratio=1
        self.var=stats_['var5']
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
        LA=math.log(beta/(1-alpha))/self.ratio
        LB=math.log((1-beta)/alpha)/self.ratio
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
    defaults=model_be.LTC_defaults
    default_biases=defaults['biases']
    default_draw_elo=defaults['draw_elo']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--alpha",help="probability of a false positve",type=float,default=0.05)
    parser.add_argument("--beta" ,help="probability of a false negative",type=float,default=0.05)
    parser.add_argument("--elo0", help="H0 (expressed in logistic elo)",type=float,default=0.0)
    parser.add_argument("--elo1", help="H1 (expressed in logistic elo)",type=float,default=5.0)
    parser.add_argument("--elo", help="actual logistic elo",type=float,required=True)
    parser.add_argument("--draw_elo", help="draw_elo",type=float,default=default_draw_elo)
    parser.add_argument("--biases", help="biases (expressed in BayesElo)",type=float,nargs='+',default=default_biases)
    parser.add_argument("--mode", help="'trinomial' or 'pentanomial'",choices=['trinomial','pentanomial'],default='pentanomial')
    args=parser.parse_args()
    alpha=args.alpha
    beta=args.beta
    elo0=args.elo0
    elo1=args.elo1
    elo=args.elo
    draw_elo=args.draw_elo
    biases=args.biases
    mode=args.mode

    print("elo0      : %.2f" % elo0)
    print("elo1      : %.2f" % elo1)
    print("elo       : %.2f" % elo)
    print("draw_elo  : %.2f" % draw_elo)
    print("biases    : %s" % biases)
    print("mode      : %s" % mode)

    alpha=0.05
    beta=0.05

    s=SPRT(alpha,beta,elo0,elo1,draw_elo,biases,mode)
    (power,expected)=s.characteristics(elo)

    print("pass probability:      %4.2f%%" % (100*power))
    print("avg running time: %10.0f" % expected)

    
