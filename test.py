import itertools
from collections import defaultdict
import random
from string import Template
from pathlib import Path

class A:
    def __init__(self, num, string, lst):
        self.num = num
        self.string = string
        self.lst = lst
    
    def get(self):
        return
    def __str__(self):
        return f"A({self.num}, {self.string}, {self.lst})"
    
    def __repr__(self):
        return self.__str__()

    def __eq__(self, value):
        if not isinstance(value, A): return False
        return (self.num == value.num and self.string == value.string and frozenset(self.lst) == frozenset(value.lst))

    def __hash__(self):
        return hash((self.num, self.string, frozenset(self.lst)))

a = [A(1,"1",[1,2]), A(2,"2",[2,3]), A(3,"3",[3,4])]
b = [A(4, "4", [4,5]), A(5, "5", [5,6]), A(6, "6", [6,7])]
c = [frozenset(a),frozenset(b)]
d = a[::-1]
e = b[::-1]
f = [frozenset(d),frozenset(e)]
print(frozenset(c) == frozenset(f))
