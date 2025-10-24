from epistemic_handler.epistemic_class import Function, Model
from functools import wraps

class Cache:
    retrive_function_cache = {}
    jp_function_cache = {}
    obs_func_cache = {}
    
    def get_rf_cache(self, seq_worlds: list[list[Function]], ts: int, func_header_id: int):
        sw = hash(tuple(frozenset([f.id for f in t]) for t in seq_worlds))
        if sw not in self.retrive_function_cache:
            self.retrive_function_cache[sw] = {}
        if ts not in self.retrive_function_cache[sw]:
            self.retrive_function_cache[sw][ts] = {}
        if func_header_id not in self.retrive_function_cache[sw][ts]:
            return None
    
    def set_rf_cache(self, seq_worlds: list[list[Function]], ts: int, func_header_id: int, func: Function | None):
        sw = hash(tuple(frozenset([f.id for f in t]) for t in seq_worlds))
        if sw not in self.retrive_function_cache:
            self.retrive_function_cache[sw] = {}
        if ts not in self.retrive_function_cache[sw]:
            self.retrive_function_cache[sw][ts] = {}
        self.retrive_function_cache[sw][ts][func_header_id] = func
    
    def get_jp_cache(self, worlds: list[list[Function]], agt_name: str):
        sw = hash(tuple(frozenset([f.id for f in t]) for t in worlds))
        if sw not in self.jp_function_cache:
            self.jp_function_cache[sw] = {}
        if agt_name not in self.jp_function_cache[sw]:
            return None
    
    def set_jp_cache(self, worlds: list[list[Function]], agt_name: str, result: list[list[Function]]):
        sw = hash(tuple(frozenset([f.id for f in t]) for t in worlds))
        if sw not in self.jp_function_cache:
            self.jp_function_cache[sw] = {}
        self.jp_function_cache[sw][agt_name] = result
    
    def get_obs_func_cache(self, world: list[Function], agt_name: str):
        sw = hash(frozenset([f.id for f in world]))
        if sw not in self.obs_func_cache:
            self.obs_func_cache[sw] = {}
        if agt_name not in self.obs_func_cache[sw]:
            return None
    
    def set_obs_func_cache(self, world: list[Function], agt_name: str, result: list[Function]):
        sw = hash(frozenset([f.id for f in world]))
        if sw not in self.obs_func_cache:
            self.obs_func_cache[sw] = {}
        self.obs_func_cache[sw][agt_name] = result

def rf_cache_decorator(func):
    @wraps(func)
    def wrapper(seq_worlds: list[list[Function]], ts: int, func_header_id: int) -> Function | None:
        result = CACHE.get_rf_cache(seq_worlds, ts, func_header_id)
        if result is not None:
            return result
        
        result = func(seq_worlds, ts, func_header_id)
        CACHE.set_rf_cache(seq_worlds, ts, func_header_id, result)
        return result
    return wrapper

def jp_cache_decorator(func):
    @wraps(func)
    def wrapper(worlds: list[list[Function]], agt_name: str, model: Model) -> list[list[Function]]:
        result = CACHE.get_jp_cache(worlds, agt_name)
        if result is not None:
            return result
        
        result = func(worlds, agt_name, model)
        CACHE.set_jp_cache(worlds, agt_name, result)
        return result
    return wrapper

def obs_func_cache_decorator(func):
    @wraps(func)
    def wrapper(self, model: Model, functions: list[Function], agent_name: str) -> list[Function]:
        result = CACHE.get_obs_func_cache(functions, agent_name)
        if result is not None:
            return result
        
        result = func(self, model, functions, agent_name)
        CACHE.set_obs_func_cache(functions, agent_name, result)
        return result
    return wrapper

CACHE = Cache()

