from __future__ import division

import math

class stats:
    """
A helper class for tracking the mean and variance
of a random variable.
"""
    def __init__(self):
        self.mean=0
        self.M2=0
        self.n=0

    def add(self,dx):
# Wikipedia!
        self.n+=1
        delta=dx-self.mean
        self.mean+=delta/self.n
        self.M2+=delta*(dx-self.mean)
        
    def params(self):
        if self.n<2:
            return (0,0) # adhoc for our application
        else:
            return self.mean,math.sqrt(self.M2/(self.n-1))

    def ci_mean(self):
        if self.n<2:
            return (0,0,0) # adhoc for our application
        else:
            return self.mean-1.96*math.sqrt(self.M2/(self.n-1))/math.sqrt(self.n),self.mean,self.mean+1.96*math.sqrt(self.M2/(self.n-1))/math.sqrt(self.n)
