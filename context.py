from __future__ import division
import sys, copy, math, random
import stats_pentanomial
import scipy

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

This implementation is based on the concept of a "context" as introduced
in section 5 of http://hardy.uhasselt.be/Fishtest/normalized_elo.pdf.
"""

bb = math.log(10) / 400


def L(x):
    if x >= 0:
        return 1 / (1 + math.exp(-bb * x))
    else:
        e = math.exp(bb * x)
        return e / (e + 1)


def score_to_elo(score):
    return -400 * math.log10(1 / score - 1)


def scale(de):
    return (4 * math.exp(-bb * de)) / (1 + math.exp(-bb * de)) ** 2


def draw_elo_calc(draw_ratio):
    return 400 * (math.log(1 / ((1 - draw_ratio) / 2.0) - 1) / math.log(10))


def ldw(belo, draw_elo, bias):
    w = L(belo - draw_elo + bias)
    l = L(-belo - draw_elo - bias)
    d = 1 - w - l
    return l, d, w


def probs_(belo, draw_elo, bias):
    ldw1 = ldw(belo, draw_elo, bias)
    ldw2 = ldw(belo, draw_elo, -bias)
    return (
        stats_pentanomial.avg([ldw1, ldw2]),
        stats_pentanomial.trinomial_to_pentanomial(ldw1, ldw2),
    )


class context:
    def __init__(self, draw_elo=None, biases=None):
        self._draw_elo = draw_elo
        self._biases = biases
        self._cache = {}
        self._ldw_cache = {}
        self.ldws = []

    def _probs(self, belo):
        """
BayesElo input!
"""
        probs3 = []
        probs5 = []
        for bias in self._biases:
            prob3, prob5 = probs_(belo, self._draw_elo, bias)
            probs3.append(prob3)
            probs5.append(prob5)
        return (stats_pentanomial.avg(probs3), stats_pentanomial.avg(probs5))

    def probs(self, elo):
        belo = self.elo_to_belo(elo)
        return self._probs(belo)

    def stats_biases(self):
        m1 = 0
        m2 = 0
        l = len(self._biases)
        for bias in self._biases:
            probs3 = ldw(0, self._draw_elo, bias)
            s = stats_pentanomial.score(probs3)
            m1 += s
            m2 += s * s
        mu = m1 / l
        sigma2 = m2 / l - mu ** 2
        return (mu - 1 / 2, sigma2)

    def elo_to_belo(self, elo):
        """
With this function logistic elo
can be converted to BayesElo for
the current context.
"""
        if elo in self._cache:
            return self._cache[elo]
        s = L(elo)
        f = lambda x: stats_pentanomial.score(self._probs(x)[1]) - s
        x, res = scipy.optimize.brentq(f, -1000, 1000, full_output=True, disp=False)
        assert res.converged
        belo = x
        self._cache[elo] = belo
        return belo

    def stats(self, elo):
        stats = {}
        probs3, probs5 = self.probs(elo)
        stats["probs3"] = probs3
        stats["probs5"] = probs5
        stats["s3"] = stats_pentanomial.score(probs3)
        stats["s5"] = stats_pentanomial.score(probs5)
        epsilon = 1e-6
        assert abs(stats["s3"] - stats["s5"]) < epsilon
        stats["var3"] = stats_pentanomial.var(probs3)
        stats["var5"] = stats_pentanomial.var(probs5)
        stats["elo"] = score_to_elo(stats["s3"])
        d = probs3[1] / (sum(probs3))
        stats["draw_ratio"] = d
        mu, sigma2 = self.stats_biases()
        stats["mu"] = mu
        stats["sigma2"] = sigma2
        stats["sigma"] = sigma2 ** 0.5
        stats["ratio"] = stats["var5"] / stats["var3"]
        v = (1 - d) / 4
        stats["ratio_predicted"] = (v - sigma2) / (v + mu ** 2)
        return stats

    def pick(self, elo):
        belo = self.elo_to_belo(elo)
        bias = random.choice(self._biases)
        if (belo, bias) in self._ldw_cache:
            ldw1, ldw2 = self._ldw_cache[(belo, bias)]
        else:
            ldw1 = ldw(belo, self._draw_elo, bias)
            ldw2 = ldw(belo, self._draw_elo, -bias)
            self._ldw_cache[(belo, bias)] = ldw1, ldw2
        i = stats_pentanomial.pick(ldw1)
        j = stats_pentanomial.pick(ldw2)
        return i, j

    def draw_elo(self):
        return self._draw_elo

    def biases(self):
        return self._biases


LTC_defaults = context(draw_elo=327, biases=[-90, 200])
