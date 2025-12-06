from dep.epistemic_class import Function, Model
import util
import cache_helper as ch
from cachetools import cached, LRUCache


def preprocess_seq_worlds(seq_worlds):
    """
    返回：
        index[header_id][t] = 在时间 t 处最近的 Function（左右都考虑）
    """
    T = len(seq_worlds)
    index = {}

    # 第一步：只记录“真实出现的位置”
    for t in range(T):
        for f in seq_worlds[t]:
            if f.value is not None:
                hid = f.header_id
                if hid not in index:
                    index[hid] = [None] * T
                index[hid][t] = f

    # 第二步：左 → 右 填充最近值
    for hid, arr in index.items():
        last = None
        for i in range(T):
            if arr[i] is not None:
                last = arr[i]
            else:
                arr[i] = last

        # 第三步：右 → 左 再填一次（补右侧最近）
        last = None
        for i in range(T - 1, -1, -1):
            if arr[i] is not None:
                last = arr[i]
            else:
                arr[i] = last

    return index


def retrieval_function(pre_index, ts: int, func_header_id: int):
    if ts == -1:
        return None

    arr = pre_index.get(func_header_id)
    if arr is None:
        return None

    if 0 <= ts < len(arr):
        return arr[ts]

    return None


# # @cached(LRUCache(maxsize=256), key=ch.freeze)
# def retrieval_function(seq_worlds: list[list[Function]], ts: int, func_header_id: int) -> Function | None:
#     if ts == -1:
#         return None
#
#     # check ts itself
#     for func in seq_worlds[ts]:
#         if func.value is not None and func.header_id == func_header_id:
#             return func
#
#     # search left hand side
#     for t in range(ts - 1, -1, -1):
#         for func in seq_worlds[t]:
#             if func.value is not None and func.header_id == func_header_id:
#                 return func
#
#     # search right hand side
#     for t in range(ts + 1, len(seq_worlds)):
#         for func in seq_worlds[t]:
#             if func.value is not None and func.header_id == func_header_id:
#                 return func
#
#     return None

# @cached(LRUCache(maxsize=512), key=ch.get_epistemic_world_key)
def get_epistemic_world(model: Model, belief_sequence: list[str], history_functions=[], ts=-1, debug=False,
                        goal_filter=True) -> list[Function]:
    """
    This is the entrance for all agents to get the epistemic world by calling the justified function based on the first element in the belief sequence.
    Here, i = history_functions[0] is equivalent to the $\vec{sigma}[0]$.
    In decentralized setting, the input ontic world needs to observe by agent i first, and determine the jp world based on the observation but not the ontic world. 
    Therefore, f([w0,...,wn]) -> f(Oi([w0,...,wn])).
    """
    if len(history_functions) == 0:
        # input [w0,...,wn], process to the observed world Oi([w0,...,wn])
        history_functions = model.get_history_functions()
        if len(belief_sequence) == 0:
            return history_functions[ts]
        history_functions = [util.OBS_FUNC[belief_sequence[0]].
                             get_observable_functions(hf,
                                                      belief_sequence[0],
                                                      model.ALL_FUNCS,
                                                      model.ontic_functions)
                             for hf in history_functions]
        # a minor change to improve the execution speed. Limit the max length of history_functions to 20
        # history_functions = history_functions[max(0,len(history_functions) - 20):]
    if len(history_functions) == 0:
        # if no history, then return empty world
        return []

    # f_sigma(Oi([w0,...,wn])) = [w'0,...,w'n]
    for i in range(len(belief_sequence)):
        history_functions = jp_function(history_functions,
                                        belief_sequence[:i + 1],
                                        model.ALL_FUNCS,
                                        model.ontic_functions)

    # return w'_ts
    return history_functions[ts]


@cached(LRUCache(maxsize=512), key=ch.freeze_ignore_all_funcs)
# def jp_function(worlds: list[list[Function]], agts: list[str], all_funcs, ontic_functions) -> list[
#     list[Function]]:
#     util.CALL_OF_JP += 1
#     index = preprocess_seq_worlds(worlds)
#
#     agt_name = agts[-1]
#     worlds2 = []
#     obs_cache = [set(
#         util.OBS_FUNC[agt_name].get_observable_functions(world, agt_name, all_funcs, ontic_functions)) for world in
#         worlds]
#
#     for t in range(len(worlds)):
#         dom_wt = [v.header_id for w in worlds for v in w]
#         dom_wt = list(set(dom_wt))
#         wt2 = set()
#
#         for v in dom_wt:
#             ltv = -1
#             for j in range(t, -1, -1):
#                 if v in [l.header_id for l in obs_cache[j]]:
#                     ltv = j
#                     break
#             e = retrieval_function(index, ltv, v)
#             if e is not None:
#                 wt2.add(e)
#         owt = obs_cache[t]
#         diff = wt2 - owt
#         for e in diff:
#             v = [f for f in owt if f.header_id == e.header_id]
#             if len(v) == 0:
#                 v = None
#             else:
#                 v = v[0]
#             owte = owt - {v} | {e}
#             oowte = util.OBS_FUNC[agt_name].get_observable_functions(list(owte),
#                                                                      agt_name,
#                                                                      all_funcs,
#                                                                      ontic_functions)
#             if v not in oowte:
#                 wt2 = wt2 - {v} | {e}
#             else:
#                 wt2 = wt2 - {e} | {v}
#         wt2 = fill_unknown(wt2, all_funcs)
#         result = util.OBS_FUNC[agt_name].post_process_jp(list(wt2), agt_name, all_funcs, ontic_functions)
#         worlds2.append(result)
#     return worlds2
def jp_function(worlds: list[list[Function]], agts: list[str], all_funcs, ontic_functions) -> list[list[Function]]:
    util.CALL_OF_JP += 1

    # Preprocess the sequence index (assumed optimized)
    index = preprocess_seq_worlds(worlds)

    agt_name = agts[-1]
    worlds2 = []

    # Precompute obs_cache: observable functions for each world
    obs_cache = [
        set(util.OBS_FUNC[agt_name].get_observable_functions(world, agt_name, all_funcs, ontic_functions))
        for world in worlds
    ]

    # Precompute sets of header_id for each obs_cache to allow O(1) lookup
    obs_cache_sets = [set(f.header_id for f in o) for o in obs_cache]

    # Precompute dom_wt: all header_ids appearing in all worlds (order does not matter)
    dom_wt = list({v.header_id for w in worlds for v in w})

    for t, owt in enumerate(obs_cache):
        wt2 = set()
        # Map header_id to Function for quick lookup
        owt_map = {f.header_id: f for f in owt}

        # Compute last observed time for each dom_wt
        for v in dom_wt:
            ltv = max((j for j in range(t, -1, -1) if v in obs_cache_sets[j]), default=-1)
            e = retrieval_function(index, ltv, v)
            if e is not None:
                wt2.add(e)

        # Process differences between wt2 and owt
        diff = wt2 - owt
        for e in diff:
            v = owt_map.get(e.header_id)
            # Construct candidate set for replacement
            candidate_set = (owt - {v} if v else owt) | {e}
            owte = util.OBS_FUNC[agt_name].get_observable_functions(
                list(candidate_set),
                agt_name,
                all_funcs,
                ontic_functions
            )

            if v not in owte:
                wt2 = (wt2 - {v} if v else wt2) | {e}
            else:
                wt2 = (wt2 - {e}) | ({v} if v else set())

        # Fill unknown functions
        wt2 = fill_unknown(wt2, all_funcs)

        # Post-process the resulting set
        result = util.OBS_FUNC[agt_name].post_process_jp(list(wt2), agt_name, all_funcs, ontic_functions)
        worlds2.append(result)

    return worlds2


# @cached(LRUCache(maxsize=256), key=ch.freeze)
def fill_unknown(functions: set[Function], all_funcs) -> set[Function]:
    fhids = [f.header_id for f in functions]
    for hid in all_funcs.header_id_add:
        if hid not in fhids:
            unknown_f = all_funcs.get_unknown_function(hid)
            if unknown_f is not None:
                functions.add(unknown_f)
    return functions
