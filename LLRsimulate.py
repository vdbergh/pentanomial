from __future__ import division
import sys,argparse

import stats_pentanomial
import SPRT_pentanomial
import context
import stats
import sprta5
import sprt

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
            return status,sp.length(),sp.LLR(),sp.results()
        if mode=='trinomial':
            sp.record(j)
        status=sp.status()
        if status!='':
            return status,sp.length(),sp.LLR(),sp.results()

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
    parser.add_argument("--verbose","-v", help="verbose",action='store_true')
    args=parser.parse_args()
    alpha=args.alpha
    beta=args.beta
    elo0=args.elo0
    elo1=args.elo1
    elo=args.elo
    mode=args.mode
    draw_elo=args.draw_elo
    biases=args.biases
    verbose=args.verbose
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
    s_elo_l=stats.stats()
    s_elo=stats.stats()
    s_elo_u=stats.stats()
    n=0
    while True:
        n+=1
        status,length,LLR,results=simulate(alpha=alpha,
                                           beta=beta,
                                           elo0=elo0,
                                           elo1=elo1,
                                           elo=elo,
                                           context=c,
                                           mode=mode)
        sp_elo=sprt.sprt(alpha=alpha,beta=beta,elo0=elo0,elo1=elo1)
        sp_elo.set_state(results)
        elo_sprt_l=sp_elo.analytics()['ci'][0]
        elo_sprt=sp_elo.analytics()['elo']
        elo_sprt_u=sp_elo.analytics()['ci'][1]
        if verbose:
            print("**** status=%s length=%d LLR=%.3f elo=%.3f[%.3f,%.3f] results=%s" % (status,
                                                                                        length,
                                                                                        LLR,
                                                                                        elo_sprt,
                                                                                        elo_sprt_l,
                                                                                        elo_sprt_u,
                                                                                        str(results)))
        s_elo_l.add(elo<=elo_sprt_l)
        s_elo.add(elo<=elo_sprt)
        s_elo_u.add(elo<=elo_sprt_u)
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
        elo_l_ci=s_elo_l.ci_mean()
        elo_ci=s_elo.ci_mean()
        elo_u_ci=s_elo_u.ci_mean()
        print(("n=%d pass=%.4f[%.4f,%.4f] length=%.1f[%.1f,%.1f]"+
              " LLR0=%.3f LLR1=%.3f l=%.4f[%.4f,%.4f]"+
              " m=%.4f[%.4f,%.4f] u=%.4f[%.4f,%.4f]") % (n,
                                                         pass_[1],pass_[0],pass_[2],
                                                         l[1],l[0],l[2],
                                                         LLR0[1],
                                                         LLR1[1],
                                                         elo_l_ci[1],elo_l_ci[0],elo_l_ci[2],
                                                         elo_ci[1],elo_ci[0],elo_ci[2],
                                                         elo_u_ci[1],elo_u_ci[0],elo_u_ci[2]))

        sys.stdout.flush()
