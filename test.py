from cachetools import cached, LRUCache

def _freeze(obj):
    """
    use for freeze the object when caching things
    """
    if isinstance(obj, list):
        return tuple(freeze(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, freeze(v)) for k, v in obj.items()))
    if isinstance(obj, set):
        return tuple(sorted(freeze(x) for x in obj))
    return obj

def freeze(*args, **kwargs):
    return _freeze((args, kwargs))

@cached(LRUCache(maxsize=256), key=_freeze)
def fibb(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibb(n - 1) + fibb(n - 2)

if __name__ == '__main__':
    a = fibb(50)
    print(a)