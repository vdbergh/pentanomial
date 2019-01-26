# pentanomial
A SPRT implementation using pentanomial frequencies and a simulation tool.

**Introduction**

In the pentanomial model an engine match consisting of paired games with reversed colors is modeled using a pentanomial distribution instead of the usual trinomial distribution. One may verify theoretically that the trinomial model overestimates elo confidence intervals. As a result it takes more effort than necessary to establish that one engine is better than another engine with a given level of significance.

This was first observed empirically by Kai Laskos on TalkChess.

***Usage***
```
$ python LLRsimulate.py -h
usage: LLRsimulate.py [-h] [--alpha ALPHA] [--beta BETA] [--elo0 ELO0]
                      [--elo1 ELO1] [--draw_elo DRAW_ELO]
                      [--biases BIASES [BIASES ...]]
                      [--mode {trinomial,pentanomial}] --elo ELO

optional arguments:
  -h, --help            show this help message and exit
  --alpha ALPHA         probability of a false positve (default: 0.05)
  --beta BETA           probability of a false negative (default: 0.05)
  --elo0 ELO0           H0 (expressed in logistic elo) (default: 0.0)
  --elo1 ELO1           H1 (expressed in logistic elo) (default: 5.0)
  --draw_elo DRAW_ELO   draw_elo (default: 327)
  --biases BIASES [BIASES ...]
                        biases (expressed in BayesElo) (default: [-90, 200])
  --mode {trinomial,pentanomial}
                        'trinomial' or 'pentanomial' (default: pentanomial)
  --elo ELO             actual logistic elo (default: None)
```
**Example**
```
$ python LLRsimulate.py --elo1 10 --elo 10 
elo0         : 0.00
elo1         : 10.00
elo          : 10.00
draw_elo     : 327.00
biases (be)  : [-90, 200]
pass_prob    : 0.950
expected     : 1809

(1, (0, 0, 0), (0, 0, 0))
(2, (1.0, 1.0, 1.0), (1880.4000000000001, 2145.0, 2409.5999999999999))
(3, (1.0, 1.0, 1.0), (673.71846669785282, 1652.0, 2630.2815333021472))
...
(58387, (0.94806409544818582, 0.94983472348296927, 0.95160535151775272), (1795.8213374456366, 1806.4565399832188, 1817.0917425208011))
(58388, (0.94806498414385776, 0.94983558265397217, 0.95160618116408657), (1795.8006071830473, 1806.4357059669828, 1817.0708047509183))
(58389, (0.94806587280911692, 0.94983644179554583, 0.95160701078197474), (1795.7783242188345, 1806.4133312781551, 1817.0483383374756))
(58390, (0.94806676144396496, 0.94983730090769181, 0.95160784037141866), (1795.7806905729024, 1806.4155163555438, 1817.0503421381852))

$ python LLRsimulate.py --elo1 10 --elo 10 --mode trinomial

elo0         : 0.00
elo1         : 10.00
elo          : 10.00
draw_elo     : 327.00
biases (be)  : [-90, 200]
pass_prob    : 0.968   <=====
expected     : 2191    <=====

(1, (0, 0, 0), (0, 0, 0))
(2, (1.0, 1.0, 1.0), (793.72000000000003, 1735.5, 2677.2799999999997))
(3, (1.0, 1.0, 1.0), (1148.1387374188928, 2187.3333333333335, 3226.5279292477744))
...
(43425, (0.96850772910603578, 0.97010938399538971, 0.97171103888474364), (2172.0810561269136, 2186.3565227403587, 2200.6319893538039))
(43426, (0.96850845373218464, 0.97011007230690827, 0.9717116908816319), (2172.0489979815034, 2186.3242757794887, 2200.5995535774741))
(43427, (0.96850917832498762, 0.97011076058672707, 0.97171234284846653), (2172.053530824413, 2186.3284822806108, 2200.6034337368087))
(43428, (0.96850990288444716, 0.97011144883484846, 0.97171299478524975), (2172.0619944668952, 2186.3366261398196, 2200.611257812744))
```
Conclusion The trinomial implementation of the SPRT overshoots the target of 95% pass probability and hence takes 20% longer to complete.

**The model**

To perform a simulation we need a method to produce realistic pentanomial frequencies. To this end we use the BayesElo model.

The BayesElo model has three inputs:

- elo
- draw_elo (a measure for the draw rate)
- advantage (the advantage for white in a position, below we call this "bias")

Hence the BayesElo model is perfect for modeling paired games with reversed colors.

One may verify theoretically that the observed difference between the trinomial and the pentanomial variance is adequately explained by the biases in the opening book: both the average bias and the variation of the bias.

Rather than modeling the biases with a continuous distribution we model them with a list of biases occuring with identical probabilities.

Realistic values are:
```
draw_elo=327
biases=[-90,200]
```
The corresponding trinomial/pentanomial probabilities in self play are
```
probs3=[0.1644, 0.6712, 0.1644]
probs5=[0.0159, 0.2189, 0.5303, 0.2189, 0.0159]
```
which are similar to what is observed in practice for LTC tests. See

https://github.com/vdbergh/compute_stats/

**The SPRT implementation**

Our SPRT implementation is in fact a GSPRT implementation. It is based on the generalized log likelihood ratio
```
(generalized)LLR=log(likelihood(MLE(params)|H1)/likelihood(MLE(params)|H0)))
```
Xiaoou Li, Jingchen Liu, and Zhiliang Ying, Generalized Sequential Probability Ratio Test for Separate Families of Hypotheses, http://stat.columbia.edu/~jcliu/paper/GSPRT_SQA3.pdf.

To apply it in our case we use the method developed in doc/MLE_multinomial.pdf.

The theoretical estimates (in sprta5.py) are made using doc/GSPRT_approximation.pdf, formula (2.1) which gives an extremely convenient asymptotic approximation to the generalized LLR.

Finally it well known that to have good agreement between the design paramaters of an SPRT and its actual characteristics one has to make a correction for discrete time (as the LLR will overshoot the boundaries). This we do also using the method pioneered by David Siegmund, Sequential Analysis, Tests and Confidende Intervals, p50.
