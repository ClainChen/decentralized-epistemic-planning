from epistemic_handler.goal_preprocesses.processes import mapf

def goal_preprocesses(base_model, agent_goal_set) -> bool | int:
    if base_model.domain_name == "mapf":
        return mapf.jump(base_model, agent_goal_set)
    return -1