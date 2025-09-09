import copy
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class A:
    value1: int = None

class Pool:
    def __init__(self):
        self.pool = []
    
    def add(self, item):
        self.pool.append(item)
    
    def get(self, index):
        return self.pool[index]

pool = Pool()
for i in range(100000):
    pool.add(A(i))

def test_copy_performance():
    sizes = [100, 1000, 10000, 100000]
    ref_times = []    # 引用赋值
    shallow_times = []  # 浅拷贝
    deep_times = []   # 深拷贝
    pool_times = []

    for size in sizes:
        a = [A(i) for i in range(size)]
        
        # 测试引用赋值
        start = time.time()
        for _ in range(1000):  # 多次测试取平均
            b = a
        ref_times.append((time.time() - start) / 1000 * 1e6)  # 微秒
        
        # 测试浅拷贝
        start = time.time()
        for _ in range(100):
            c = a[:]
        shallow_times.append((time.time() - start) / 100 * 1e6)  # 微秒

        # 测试对象池拷贝
        start = time.time()
        for _ in range(10 if size <= 10000 else 1):  # 大数组减少测试次数
            d = [pool.get(p.value1) for p in a]
        pool_times.append((time.time() - start) / (10 if size <= 10000 else 1) * 1e6)
        
        # 测试深拷贝
        start = time.time()
        for _ in range(10 if size <= 10000 else 1):  # 大数组减少测试次数
            e = copy.deepcopy(a)
        deep_times.append((time.time() - start) / (10 if size <= 10000 else 1) * 1e6)
    
    return sizes, ref_times, shallow_times, pool_times, deep_times

# 运行测试
sizes, ref_times, shallow_times, pool_times, deep_times = test_copy_performance()

# 输出结果
print("大小\t引用赋值(μs)\t浅拷贝(μs)\t对象池(μs)\t深拷贝(μs)")
for i in range(len(sizes)):
    print(f"{sizes[i]}\t{ref_times[i]:.3f}\t\t{shallow_times[i]:.3f}\t\t{pool_times[i]:.3f}\t\t{deep_times[i]:.3f}")

