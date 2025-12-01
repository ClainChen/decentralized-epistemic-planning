from epistemic_handler.epistemic_class import Function, Model, Condition
import copy
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

def get_epistemic_world(model: Model, belief_sequence: list[str], history_functions=[], ts=-1, debug=False, goal_filter=True) -> list[Function]:
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
        history_functions = jp_function(history_functions, belief_sequence[:i+1], model, debug=debug, goal_filter=goal_filter)
    
    # return w'_ts
    return history_functions[ts]


def get_unfiltered_st(world_seq: list[list[Function]]) -> list[Function]:
    """
    get the epistemic world from the given function sequence\n
    this usually use when checking the epistemic condition and generating the virtual world\n
    """
    if len(world_seq) == 0:
        return []
    
    world = []
    headers = set()
    for functions in reversed(world_seq):
        for func in functions:
            if func.header_id not in headers:
                headers.add(func.header_id)
                world.append(func)
    return world


def jp_function(worlds: list[list[Function]], agts: list[str], model: Model, debug=False, goal_filter=False) -> list[list[Function]]:
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
        # if debug and t == len(worlds) - 1:
        #     print(agts)
        #     print('world:')
        #     for f in worlds[t]:
        #         print(f)
        #     print('wt2:')
        #     for f in wt2:
        #         print(f)
        #     print('owt:')
        #     for f in owt:
        #         print(f)
        #     print('diff:')
        #     for f in diff:
        #         print(f)
        #     print("=====")
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
        # if goal_filter:
        #     goal_signal_filter(wt2, owt, agts, model, t)
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

def goal_signal_filter(functions: list[Function], obs_funcs: list[Function], agts: list[str], model: Model, t: int):
    if len(model.history) == 0:
        goal_signal = None
    else:
        goal_signal = model.history[t]['signal'] if t < len(model.history) else {agt.name: agt.complete_signal for agt in model.agents}
    agent_name = agts[0]
    poss_goals = model.get_agent_by_name(agent_name).all_possible_goals

    for poss_goal in poss_goals:
        for a, gs in poss_goal.items():
            for g in gs:
                needs_filter = len(g.belief_sequence) == len(agts) - 1
                if not needs_filter:
                    continue
                obs_f = model.ALL_FUNCS.get_function_with_cond(g)
                for i, j in zip(g.belief_sequence, agts[1:]):
                    if i != j:
                        needs_filter = False
                        break
                if (needs_filter and
                    goal_signal is not None and 
                    obs_f not in obs_funcs and 
                    util.check_regular_condition(g, functions) != goal_signal[a]):
                    # remove those functions that do not satisfy the goal signal
                    # if t >= len(model.hstory):
                    #     print(agts, g.belief_sequence)
                    #     print(obs_f)
                    #     for f in functions:
                    #         print(f)
                    #     print("=====")
                    target = util.get_function_with_name_and_params(functions, obs_f.name, obs_f.parameters)
                    if target is not None:
                        functions.remove(target)

# def get_epistemic_world(model: Model, belief_sequence: list[str], history_functions=[], goal_filter=False) -> list[Function]:
#     from util import OBS_FUNC
#     """
#     if belief_sequence = [a,b,c], history = [S0, S1, ..., Sn]
#     output: st' = st'' / ( Oc(st'') / Oc(st) )
#     st = fb(fa(St))
#     st'' = fc(fb(fa(St)))
#     """
#     if len(history_functions) == 0:
#         history_functions = model.get_history_functions()
#         if len(belief_sequence) == 0:
#             return history_functions[-1]
#         history_functions = [util.OBS_FUNC[belief_sequence[0]].
#                          get_observable_functions(model, hf, belief_sequence[0]) 
#                          for hf in history_functions]
#     if len(history_functions) == 0:
#         return []
#     if len(belief_sequence) == 0:
#         return history_functions[-1]

#     # st''
#     history_beliefs = [get_functions_with_belief_sequence(functions, belief_sequence, model) for functions in history_functions]
#     st2 = get_unfiltered_st(history_beliefs)

#     # st
#     st = get_epistemic_world(model, belief_sequence[:-1], history_functions)

#     # Oi(st'')
#     last_agt = belief_sequence[-1]
#     Oi_st2 = set(OBS_FUNC[last_agt].get_observable_functions(model, st2, last_agt))
    
#     # Oi(st)
#     Oi_st = set(OBS_FUNC[last_agt].get_observable_functions(model, st, last_agt))

#     return list(set(st2).difference(Oi_st2.difference(Oi_st)))

def get_functions_with_belief_sequence(functions: list[Function], belief_sequence: list[str], model: Model) -> list[Function]:
    from util import OBS_FUNC
    if len(belief_sequence) == 0:
        return functions
    ontic_functions = functions
    for agent_name in belief_sequence:
        ontic_functions = OBS_FUNC[agent_name].get_observable_functions(model, ontic_functions, agent_name)
    return ontic_functions