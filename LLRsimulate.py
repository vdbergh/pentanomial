from __future__ import division
import sys,argparse

import stats_pentanomial
import SPRT_pentanomial
import context
import stats
import sprta5

def simulate(alpha=0.05,beta=0.05,elo0=None,elo1=None,elo=None, context=None, mode='pentanomial'):
    """
    We simulate the test H0:elo==elo0 versus H1:elo==elo1. All elo inputs are in logistic elo.
"""
    sp=SPRT_pentanomial.SPRT(alpha=alpha,beta=beta,elo0=elo0,elo1=elo1,mode=mode)
    assert(mode in ('trinomial','pentanomial'))
    while True:
        i,j=context.pick(elo)
        if mode=='trinomial':
            sp.record(i)
        else:
            sp.record(i+j)
        status=sp.status()
        if status!='':
            return status,sp.length(),sp.LLR()
        if mode=='trinomial':
            sp.record(j)
        status=sp.status()
        if status!='':
            return status,sp.length(),sp.LLR()

if __name__=='__main__':
    defaults=context.LTC_defaults
    default_biases=defaults.biases()
    default_draw_elo=defaults.draw_elo()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--alpha",help="probability of a false positve",type=float,default=0.05)
    parser.add_argument("--beta" ,help="probability of a false negative",type=float,default=0.05)
    parser.add_argument("--elo0", help="H0 (expressed in logistic elo)",type=float,default=0.0)
    parser.add_argument("--elo1", help="H1 (expressed in logistic elo)",type=float,default=5.0)
    parser.add_argument("--draw_elo", help="draw_elo",type=float,default=default_draw_elo)
    parser.add_argument("--biases", help="biases (expressed in BayesElo)",type=float,nargs='+',default=default_biases)
    parser.add_argument("--mode", help="'trinomial' or 'pentanomial'",choices=['trinomial','pentanomial'],default='pentanomial')
    parser.add_argument("--elo", help="actual logistic elo",type=float,required=True)
    args=parser.parse_args()
    alpha=args.alpha
    beta=args.beta
    elo0=args.elo0
    elo1=args.elo1
    elo=args.elo
    mode=args.mode
    draw_elo=args.draw_elo
    biases=args.biases
    c=context.context(draw_elo,biases)
    sp=sprta5.SPRT(alpha=alpha,beta=beta,elo0=elo0,elo1=elo1,context=c,mode=mode)
    pass_prob,expected_length=sp.characteristics(elo)
    print("elo0         : %.2f" % elo0)
    print("elo1         : %.2f" % elo1)
    print("elo          : %.2f" % elo)
    print("draw_elo     : %.2f" % draw_elo)
    print("biases (be)  : %s" % (str(biases)))
    print("pass_prob    : %.3f" % pass_prob)
    print("expected     : %.0f" % expected_length)
    print("")
    s_pass=stats.stats()
    s_length=stats.stats()
    s_LLR_H0=stats.stats()
    s_LLR_H1=stats.stats()
    n=0
    while True:
        n+=1
        status,length,LLR=simulate(alpha=alpha,beta=beta,elo0=elo0,elo1=elo1,elo=elo,context=c,mode=mode)
        s_length.add(length)
        if status=='H1':
            s_pass.add(1.0)
            s_LLR_H1.add(LLR)
        else:
            s_pass.add(0.0)
            s_LLR_H0.add(LLR)
        pass_=s_pass.ci_mean()
        l=s_length.ci_mean()
        LLR0=s_LLR_H0.ci_mean()
        LLR1=s_LLR_H1.ci_mean()
        print("n=%d pass=%.4f[%.4f,%.4f] lengths=%.1f[%.1f,%.1f] LLR0=%.3f[%.3f,%.3f] LLR1=%.3f[%.3f,%.3f]" % (n,
                                                                                                              pass_[1],pass_[0],pass_[2],
                                                                                                              l[1],l[0],l[2],
                                                                                                              LLR0[1],LLR0[0],LLR0[2],
                                                                                                              LLR1[1],LLR1[0],LLR1[2]))
        sys.stdout.flush()
