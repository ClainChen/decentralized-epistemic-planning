import itertools
from collections import defaultdict
import random
from string import Template
from pathlib import Path
import copy

class A:
    def __init__(self, lst):
        self.lst = lst
    
    def __repr__(self):
        return str(self.lst)

    def __str__(self):
        return str(self.lst)

    def __hash__(self):
        return hash(tuple(set(self.lst)))
    
    def __eq__(self, value):
        if not isinstance(value, A): return False
        return frozenset(self.lst) == frozenset(value.lst)

a = [A([1,3,5,7]), A([2,4,6,8])]
b = frozenset(a)
a[0].lst[2] = 9
print(a)
print(b)


