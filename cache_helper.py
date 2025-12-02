def _freeze(obj):
    """
    use for freeze the object when caching things
    """
    if isinstance(obj, list):
        return tuple(_freeze(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((_freeze(k), _freeze(v)) for k, v in obj.items()))
    if isinstance(obj, set):
        return tuple(sorted(_freeze(x) for x in obj))
    # 处理其他可迭代对象（除了字符串）
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            return tuple(_freeze(x) for x in obj)
        except TypeError:
            # 如果不可迭代，返回原对象
            pass
    return obj

def freeze(*args, **kwargs):
    return _freeze((args, kwargs))