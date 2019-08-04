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

if __name__=='__main__':
    import random
    N=100000000
    A=1
    B=1000000
    mu_,sigma_=B+A/2.0,A/12**.5
    s=stats()
    for _ in range(0,N):
        r=A*random.random()+B
        s.add(r)
    print('Confidence interval for true mean=%f' % mu_)
    print(s.ci_mean())
    print('Sample mean and stdev:')
    mu,sigma=s.params()
    print(mu,sigma)
    print('Relative error:')
    print((mu-mu_)/mu_,(sigma-sigma_)/sigma_)
