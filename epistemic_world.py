from dep.epistemic_class import Function, Model
import util


def retrieval_function(seq_worlds: list[list[Function]], ts: int, func_header_id: int) -> Function | None:
    if ts == -1:
        return None

    # check ts itself
    for func in seq_worlds[ts]:
        if func.value != None and func.header_id == func_header_id:
            return func

    # search left hand side
    for t in range(ts - 1, -1, -1):
        for func in seq_worlds[t]:
            if func.value != None and func.header_id == func_header_id:
                return func

    # search right hand side
    for t in range(ts + 1, len(seq_worlds)):
        for func in seq_worlds[t]:
            if func.value != None and func.header_id == func_header_id:
                return func

    return None


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
                             get_observable_functions(model, hf, belief_sequence[0])
                             for hf in history_functions]
    if len(history_functions) == 0:
        # if no history, then return empty world
        return []

    # f_sigma(Oi([w0,...,wn])) = [w'0,...,w'n]
    for i in range(len(belief_sequence)):
        history_functions = jp_function(history_functions, belief_sequence[:i + 1], model, debug=debug,
                                        goal_filter=goal_filter)

    # return w'_ts
    return history_functions[ts]


def jp_function(worlds: list[list[Function]], agts: list[str], model: Model, debug=False, goal_filter=False) -> list[
    list[Function]]:
    from util import OBS_FUNC
    agt_name = agts[-1]
    worlds2 = []
    obs_cache = [set(OBS_FUNC[agt_name].get_observable_functions(model, world, agt_name)) for world in worlds]

    for t in range(len(worlds)):
        dom_wt = [v.header_id for w in worlds for v in w]
        dom_wt = list(set(dom_wt))
        wt2 = set()

        for v in dom_wt:
            ltv = -1
            for j in range(t, -1, -1):
                if v in [l.header_id for l in obs_cache[j]]:
                    ltv = j
                    break
            e = retrieval_function(worlds, ltv, v)
            if e is not None:
                wt2.add(e)
        owt = obs_cache[t]
        diff = wt2 - owt
        for e in diff:
            v = [f for f in owt if f.header_id == e.header_id]
            if len(v) == 0:
                v = None
            else:
                v = v[0]
            owte = owt - {v} | {e}
            oowte = OBS_FUNC[agt_name].get_observable_functions(model, list(owte), agt_name)
            if v not in oowte:
                wt2 = wt2 - {v} | {e}
            else:
                wt2 = wt2 - {e} | {v}
        wt2 = fill_unknwon(wt2, model)
        worlds2.append(list(wt2))
    return worlds2


def fill_unknwon(functions: set[Function], model: Model) -> set[Function]:
    fhids = [f.header_id for f in functions]
    for hid in model.ALL_FUNCS.header_id_add:
        if hid not in fhids:
            unknown_f = model.ALL_FUNCS.get_unknown_function(hid)
            if unknown_f is not None:
                functions.add(unknown_f)
    return functions
