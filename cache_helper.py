from functools import lru_cache

_HASHABLE = (int, float, str, bytes, bool, type(None))


def freeze_ignore_all_funcs(worlds, agts, all_funcs, ontic_functions):
    return tuple(tuple(f) for f in worlds), tuple(agts), tuple(ontic_functions)


def freeze_lst_to_tuple(functions):
    return tuple(functions)


from cachetools import cached, LRUCache


# Custom key generator
def get_epistemic_world_key(model, belief_sequence, history_functions=[], ts=-1, debug=False, goal_filter=True):
    # Freeze lists into tuples for hashability
    def freeze(obj):
        if isinstance(obj, list):
            return tuple(freeze(x) for x in obj)
        return obj

    if len(history_functions) > 0:
        return model.bfs_key(), tuple(belief_sequence), tuple(tuple(f) for f in history_functions), ts
    else:
        return model.bfs_key(), tuple(belief_sequence), (), ts


class BfsCache:
    def __init__(self):
        self.cache: dict[int, int] = {}

    def add_cache(self, bfs_node, l=0):
        for model, length in zip(bfs_node.model[1:],
                                 range(len(bfs_node.model) - 2, 0, -1)):
            key = model.bfs_key()
            if key not in self.cache or self.cache[key] > length + l:
                self.cache[key] = length + l

    def get_cache(self, model):
        key = model.bfs_key()
        if key in self.cache:
            return self.cache[key]
        return None


BFS_CACHE = BfsCache()
