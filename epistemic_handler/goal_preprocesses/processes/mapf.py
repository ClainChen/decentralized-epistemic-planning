import itertools
import util

def jump(base_model, agent_goal_set) -> bool:
    # agent only care about their own position
    for agt_name, goals in agent_goal_set.items():
        for goal in goals:
            func = base_model.ALL_FUNCS.get_function_with_cond(goal)
            if func.name == "agent_at" and func.parameters['?a'] != agt_name:
                return True
    # easy conflict check
    goal_set = {goal for goals in agent_goal_set.values() for goal in goals}
    for goal1, goal2 in itertools.combinations(goal_set, 2):
        if not util.RULES.check_valid_pair(goal1, goal2, base_model):
            return True
    return False