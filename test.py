from dataclasses import dataclass, field

@dataclass
class Test:
    num:int = 1
    lst:list = field(default_factory=lambda: ['a', 'b'])

class Test2:
    def __init__(self):
        test: Test = None

def append_layer_1(cls:Test):
    append_layer_2(cls)

def append_layer_2(cls:Test):
    cls.num += 1
    cls.lst.append('c')

t = Test()
t2 = Test2()

t2.test = t

append_layer_1(t2.test)

print(t)