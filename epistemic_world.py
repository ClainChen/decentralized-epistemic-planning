from epistemic_handler.epistemic_class import Function, Model


def retrive_function(seq_worlds: list[list[Function]], ts: int, func_header_id: int) -> Function | None:
    if ts == -1:
        return None
    
    # check ts itself
    for func in seq_worlds[ts]:
        if func.header_id == func_header_id:
            return func
    
    # search left hand side
    for t in range(ts - 1, -1, -1):
        for func in seq_worlds[t]:
            if func.header_id == func_header_id:
                return func
    
    # search right hand side
    for t in range(ts + 1, len(seq_worlds)):
        for func in seq_worlds[t]:
            if func.header_id == func_header_id:
                return func
    
    return None

def get_epistemic_world(model: Model, belief_sequence: list[str], history_functions=[], ts=-1) -> list[Function]:
    if len(history_functions) == 0:
        history_functions = model.get_history_functions()
    if len(history_functions) == 0:
        return []
    
    # [a,b,c] -> f_c(f_b(f_a(ws)))
    # [] -> ws[-1]

    for agt in belief_sequence:
        history_functions = jp_function(history_functions, agt, model)

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



def jp_function(worlds: list[list[Function]], agt_name: str, model: Model) -> list[list[Function]]:
    from util import OBS_FUNC
    worlds2 = []
    obs_cache = [set(OBS_FUNC[agt_name].get_observable_functions(model, worlds[t], agt_name)) for t in range(len(worlds))]
    # dom_wt = [v.header_id for world in worlds for v in world]
    # dom_wt = list(set(dom_wt))

    for t in range(len(worlds)):
        dom_wt = [v.header_id for v in worlds[t]]
        dom_wt = list(set(dom_wt))
        wt2 = set()
        for v in dom_wt:
            ltv = -1
            for j in range(t, -1, -1):
                if v in [l.header_id for l in obs_cache[j]]:
                    ltv = j
                    break

            e = retrive_function(worlds, ltv, v)
            if e is not None:
                wt2.add(e)
        
        owt2 = set(OBS_FUNC[agt_name].get_observable_functions(model, list(wt2), agt_name))
        owt = obs_cache[t]
        wt1 = wt2 - (owt2 - owt)
        if t == len(worlds) - 1:
            print("\nwt2")
            for f in wt2:
                print(f)
            print("\nowt2")
            for f in owt2:
                print(f)
            print("\nowt")
            for f in owt:
                print(f)
            print("\nwt1")
            for f in wt1:
                print(f)
            print("------")
        worlds2.append(list(wt1))
    
    return worlds2

# def get_epistemic_world(model: Model, belief_sequence: list[str], history_functions=[]) -> list[Function]:
#     from util import OBS_FUNC
#     """
#     if belief_sequence = [a,b,c], history = [S0, S1, ..., Sn]
#     output: st' = st'' / ( Oc(st'') / Oc(st) )
#     st = fb(fa(St))
#     st'' = fc(fb(fa(St)))
#     """
#     if len(history_functions) == 0:
#         history_functions = model.get_history_functions()
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