from __future__ import division
import random


def var(probs):
    l = len(probs)
    s = score(probs) * (l - 1) / 2
    v = sum([probs[i] * (i / 2 - s) ** 2 for i in range(0, l)])
    return v / ((l - 1) / 2)


def score(probs):
    l = len(probs)
    s = sum([probs[i] * (i / 2) for i in range(0, l)])
    return s / ((l - 1) / 2)


def add(lists):
    l = len(lists)
    l1 = len(lists[0])
    s = l1 * [0]
    for i in range(0, l1):
        s[i] = 0
        for j in range(0, l):
            s[i] += lists[j][i]
    return s


def mult_scalar(scalar, l):
    return [scalar * ll for ll in l]


def avg(lists):
    l = add(lists)
    return mult_scalar(1 / len(lists), l)


def pick(probs):
    s = random.random()
    p = 0
    for i in range(0, len(probs)):
        pp = probs[i]
        p += pp
        if p >= s:
            return i


def trinomial_to_pentanomial(ldw1, ldw2):
    p = 5 * [0]
    for i in range(0, 3):
        for j in range(0, 3):
            p[i + j] += ldw1[i] * ldw2[j]
    return p
