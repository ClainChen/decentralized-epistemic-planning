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

from tqdm import tqdm
import time

# 基本用法
for i in tqdm(range(100)):
    time.sleep(0.1)  # 模拟任务

# 自定义描述
with tqdm(range(100), desc="处理进度") as pbar:
    for i in pbar:
        time.sleep(0.1)
        pbar.set_postfix({"当前值": i})  # 添加额外信息