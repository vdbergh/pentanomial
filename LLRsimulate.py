from __future__ import division
import sys,argparse

import stats_pentanomial
import SPRT_pentanomial
import model_be
import stats
import sprta5

def simulate(probs=None,alpha=0.05,beta=0.05,elo0=None,elo1=None,elo=None, mode='pentanomial'):
    """
    We simulate the test H0:elo==elo0 versus H1:elo==elo1. All elo inputs are in logistic elo.
"""
    sp=SPRT_pentanomial.SPRT(alpha=alpha,beta=beta,elo0=elo0,elo1=elo1,mode=mode)
    while True:
        r=stats_pentanomial.pick(probs)
        sp.record(r)
        status=sp.status()
        if status!='':
            return status,sp.length()

if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--alpha",help="probability of a false positve",type=float,default=0.05)
    parser.add_argument("--beta" ,help="probability of a false negative",type=float,default=0.05)
    parser.add_argument("--elo0", help="H0 (expressed in logistic elo)",type=float,default=0.0)
    parser.add_argument("--elo1", help="H1 (expressed in logistic elo)",type=float,default=5.0)
    parser.add_argument("--mode", help="'trinomial' or 'pentanomial'",choices=['trinomial','pentanomial'],default='pentanomial')
    parser.add_argument("--elo", help="actual logistic elo",type=float,required=True)
    args=parser.parse_args()
    alpha=args.alpha
    beta=args.beta
    elo0=args.elo0
    elo1=args.elo1
    elo=args.elo
    mode=args.mode
    draw_elo=model_be.LTC_defaults['draw_elo']
    biases=model_be.LTC_defaults['biases']
    belo=model_be.elo_to_belo(elo=elo,draw_elo=draw_elo,biases=biases)
    probs=model_be.probs(belo,draw_elo,biases)[1 if mode=='pentanomial' else 0]
    assert(abs(stats_pentanomial.score(probs)-model_be.L(elo))<1e-3)
    var=stats_pentanomial.var(probs)
    sp=sprta5.SPRT(alpha=alpha,beta=beta,elo0=elo0,elo1=elo1,var=var)
    pass_prob,expected_length=sp.characteristics(elo)
    print("elo0         : %.2f" % elo0)
    print("elo1         : %.2f" % elo1)
    print("elo          : %.2f" % elo)
    print("bayes_elo    : %.2f" % belo)
    print("draw_elo     : %.2f" % draw_elo)
    print("biases (be)  : %s" % (str(biases)))
    print("probs        : %s" % (str(probs)))
    print("var          : %.4f" % var)
    print("pass_prob    : %.3f" % pass_prob)
    print("expected     : %.0f" % expected_length)
    print("")
    s_pass=stats.stats()
    s_length=stats.stats()
    n=0
    while True:
        n+=1
        status,length=simulate(probs=probs,alpha=alpha,beta=beta,elo0=elo0,elo1=elo1,elo=elo,mode=mode)
        s_length.add(length)
        if status=='H1':
            s_pass.add(1.0)
        else:
            s_pass.add(0.0)
        print(n,s_pass.ci_mean(), s_length.ci_mean())
        sys.stdout.flush()
