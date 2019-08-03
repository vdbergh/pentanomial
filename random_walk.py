from __future__ import division

import math,sys

def _aux(x):
    """
Evaluation of (exp(x)-1-x)/x
with 8 significant digits.
"""
    if abs(x)<1e-4:
        return x/2+x**2/6+x**3/24+x**4/120
    else:
        return (math.exp(x)-1-x)/x

def _aux2(x):
    """
Evaluation of (exp(x)-1-x-x^2/2)/x^2
with 7 significant digits.
"""
    if abs(x)<1e-3:
        return x/6+x**2/24+x**3/120+x**4/720
    else:
        return (math.exp(x)-1-x-x*x/2)/(x*x)

def _paux2(x):
    """
Robust evaluation of the derivative of
_aux2(x).
"""
    if abs(x)<1e-4:
        return 1/6+x/12+x**2/40+x**3/180
    else:
        return (_aux(x)-2*_aux2(x))/x

class RandomWalk:
    def __init__(self,a,b,jumps):
        """
Jumps is a probability density consisting
of a list of tuples (jump,probablity).
"""
        self.a=a
        self.b=b
        self.jumps=jumps
        self.mu=sum([prob*jump for jump,prob in self.jumps])
        self.m2=sum([prob*jump*jump for jump,prob in self.jumps])
        
    def _e(self):
        mu=self.mu
        m2=self.m2
        e=-2/m2
        while True:
            t=m2/2+sum([prob*jump*jump*_aux2(e*mu*jump) for jump,prob in self.jumps])
            g=1+e*t
            if abs(g*mu)<1e-12: # adhoc stopping condition
                break
            tp=sum([prob*jump*jump*jump*mu*_paux2(e*mu*jump) for jump,prob in self.jumps]) 
            gp=e*tp+t
            e-=g/gp
        return e

    def characteristics(self):
        a=self.a
        b=self.b
        mu=self.mu
        e=self._e()
        h=e*mu
        prob_H1=-(a+a*_aux(h*a))/(b-a+b*_aux(h*b)-a*_aux(h*a))
        E=e*a*b*(b/2-a/2+b*_aux2(h*b)-a*_aux2(h*a))/(b-a+b*_aux(h*b)-a*_aux(h*a)) 
        return prob_H1,E
                 

    
