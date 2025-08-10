import itertools
from collections import defaultdict
import random

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

a = list(range(10))
random.shuffle(a)
dic = defaultdict(list)
for i in a:
    dic[i % 10].append(i)
b = itertools.product(*dic.values())
b = [set(i) for i in b]
print(b)