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

def get_cross_subsets(nested_list):
    # 为每个子列表生成选择（包括空选择）
    choices = [[[]] + [[item] for item in sublist] for sublist in nested_list]
    
    # 生成所有组合，展平并过滤空列表
    return [list(itertools.chain.from_iterable(combo)) 
            for combo in itertools.product(*choices) 
            if any(combo)]

a = [[1],[['a','b'],['c','d']]]
b = [[1,2],[3,4],[5,6]]
print(get_cross_subsets(b))
# b = [list(e) for e in itertools.product(*a)]
# print(b)


