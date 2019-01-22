from __future__ import division
import sys,copy,math,random
import stats_pentanomial
import brentq
"""
Here we model pentanomial probabilities using the BayesElo model.

The difference between the trinomial and the pentanomial variance
is caused by the biases in the opening book: both the average
bias and the variation of the bias.

Rather than modeling the biases with a continuous distribution we
model them with a list of biases occuring with identical
probabilities.

Relistic values are:

draw_elo=327
biases=[-90,200]

The corresponding trinomial/pentanomial probabilities in self play are

probs3=[0.16439305913734648, 0.6712138817253069, 0.16439305913734648]
probs5=[0.015923689309074094, 0.21891322446235123, 0.5303261724571493, 0.21891322446235123, 0.015923689309074094]

which are similar to what is observed in practice for LTC tests. See

https://github.com/vdbergh/compute_stats/

The ratio var5/var3 is equal to 0.86.
"""

LTC_defaults={'draw_elo':327,'biases':[-90,200]}

bb=math.log(10)/400

def L(x):
    if x>=0:
        return 1/(1+math.exp(-bb*x))
    else:
        e=math.exp(bb*x)
        return e/(e+1)

def score_to_elo(score):
    return -400*math.log10(1/score-1)

def scale(de):
    return (4*math.exp(-bb*de))/(1+math.exp(-bb*de))**2

def draw_elo_calc(draw_ratio):
    return 400*(math.log(1/((1-draw_ratio)/2.0)-1)/math.log(10))

def ldw(belo,draw_elo,bias):
    w=L(belo-draw_elo+bias)
    l=L(-belo-draw_elo-bias)
    d=1-w-l
    return l,d,w

def probs_(belo,draw_elo,bias):
    ldw1=ldw(belo,draw_elo,bias)
    ldw2=ldw(belo,draw_elo,-bias)
    return (stats_pentanomial.avg([ldw1,ldw2]),
            stats_pentanomial.trinomial_to_pentanomial(ldw1,ldw2))

def probs(belo,draw_elo,biases):
    probs3=[]
    probs5=[]
    for bias in biases:
        prob3,prob5=probs_(belo,draw_elo,bias)
        probs3.append(prob3)
        probs5.append(prob5)
    return(stats_pentanomial.avg(probs3),stats_pentanomial.avg(probs5))

def stats_biases(draw_elo,biases):
    m1=0
    m2=0
    l=len(biases)
    for bias in biases:
        probs3=ldw(0,draw_elo,bias)
        s=stats_pentanomial.score(probs3)
        m1+=s
        m2+=s*s
    mu=m1/l
    sigma2=m2/l-mu**2
    return (mu-1/2,sigma2)

def elo_to_belo(elo,draw_elo,biases):
    """
The other functions in this package
all take BayesElo as input.

With this function logistic elo
can be converted to BayesElo.
"""
    s=L(elo)
    f=lambda x:stats_pentanomial.score(probs(x,draw_elo,biases)[1])-s
    res=brentq.brentq(f,-1000,1000)
    assert(res['converged'])
    return res['x0']

def stats_logistic(belo,draw_elo,biases):
    stats={}
    probs3,probs5=probs(belo,draw_elo,biases)
    stats['probs3']=probs3
    stats['probs5']=probs5
    stats['s3']=stats_pentanomial.score(probs3)
    stats['s5']=stats_pentanomial.score(probs5)
    epsilon=1e-6
    assert(abs(stats['s3']-stats['s5'])<epsilon)
    stats['var3']=stats_pentanomial.var(probs3)
    stats['var5']=stats_pentanomial.var(probs5)
    stats['elo']=score_to_elo(stats['s3'])
    d=probs3[1]/(sum(probs3))
    stats['draw_ratio']=d
    mu,sigma2=stats_biases(draw_elo,biases)
    stats['mu']=mu
    stats['sigma2']=sigma2
    stats['sigma']=sigma2**.5
    stats['ratio']=stats['var5']/stats['var3']
    v=(1-d)/4
    stats['ratio_predicted']=(v-sigma2)/(v+mu**2)
    return stats

def pick(belo,draw_elo,biases):
    bias=random.choice(biases)
    ldw1=ldw(belo,draw_elo,bias)
    ldw2=ldw(belo,draw_elo,-bias)
    i=stats_pentanomial.pick(ldw1)
    j=stats_pentanomial.pick(ldw2)
    return i,j
