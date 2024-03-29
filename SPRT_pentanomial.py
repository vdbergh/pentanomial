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

    def __init__(
        self,
        alpha=0.05,
        beta=0.05,
        elo0=0,
        elo1=5,
        mode="pentanomial",
        elo_model="logistic",
    ):
        self.elo0 = elo0
        self.elo1 = elo1
        assert elo_model in ("logistic", "normalized")
        assert mode in ("trinomial", "pentanomial")
        if mode == "pentanomial":
            self.results_ = 5 * [0]
        else:
            self.results_ = 3 * [0]
        self.LA = math.log(beta / (1 - alpha))
        self.LB = math.log((1 - beta) / alpha)
        self.status_ = ""
        self.LLR_ = 0.0
        self.min_LLR = 0.0
        self.max_LLR = 0.0
        self.sq0 = 0.0
        self.sq1 = 0.0
        self.o0 = 0.0
        self.o1 = 0.0
        self.elo_model = elo_model

    def record(self, result):
        if self.status_ != "":
            return
        self.results_[result] += 1
        if self.elo_model == "logistic":
            self.LLR_ = LLRcalc.LLR_logistic(self.elo0, self.elo1, self.results_)
        else:
            self.LLR_ = LLRcalc.LLR_normalized(self.elo0, self.elo1, self.results_)

        # Dynamic overshoot correction using
        # Siegmund - Sequential Analysis - Corollary 8.33.
        if self.LLR_ > self.max_LLR:
            self.sq1 += (self.LLR_ - self.max_LLR) ** 2
            self.max_LLR = self.LLR_
            self.o1 = self.sq1 / self.LLR_ / 2
        if self.LLR_ < self.min_LLR:
            self.sq0 += (self.LLR_ - self.min_LLR) ** 2
            self.min_LLR = self.LLR_
            self.o0 = -self.sq0 / self.LLR_ / 2

        if self.LLR_ > self.LB - self.o1:
            self.status_ = "H1"
        elif self.LLR_ < self.LA + self.o0:
            self.status_ = "H0"

    def status(self):
        return self.status_

    def length(self):
        l = len(self.results_)
        return ((l - 1) / 2) * sum(self.results_)

    def LLR(self):
        return self.LLR_

    def results(self):
        return self.results_
