from collections import defaultdict

import numpy as np
import pandas as pd


class Count:
    def __init__(self):
        self.count = defaultdict(lambda: 0)

    def __call__(self, s):
        for w in frozenset(s):
            self.count[w] += 1
        return 0

    def finalize(self, n):
        n = float(n)
        return {w: c/n for w, c in self.count.iteritems()}


class ReprCalcer:
    def __init__(self, w2v, wp):
        self.w2v = w2v
        self.WP = wp
        self.zero = [np.array([0]*100)]
        self.alfa = 0.0005

    def __call__(self, words):
        s = np.sum([self.alfa / (self.alfa + self.WP[w]) * self.w2v.wv[w] for w in words if w in self.w2v.wv] or self.zero, axis=0)
        norm = sum([self.alfa / (self.alfa + self.WP[w]) for w in words if w in self.w2v.wv] or [1.0])
        vect = s / norm
        # vect = vect / (np.linalg.norm(vect) or 1.0)
        return vect
