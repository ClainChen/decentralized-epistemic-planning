import logging
from epistemic_handler.file_parser import *
from epistemic_handler.epistemic_class import Model, ProblemType, Condition

class ModelChecker:
    """
    This class use to check the validity of the given parsing domain and problem.
    """
    def __init__(self, domain, problem):
        self.domain: ParsingDomain = domain
        self.problem: ParsingProblem = problem
    
    def check_validity(self) -> bool:
        """
        check the validity based on given process:\n
        1. name: problem & domain, wrong function name usage
        2. agent
        3. object
        4. initial state validity
        """
        result = True
        util.LOGGER.info("checking the validity...")

        # check the domain name validity
        if self.domain.name != self.problem.domain_name:
            util.LOGGER.debug(f"The domain name is different from the problem's domain name")
        # gather all valid names
        valid_names = self.problem.agents[:]
        for object_names in self.problem.objects.values():
            valid_names += object_names
        valid_names.append('unknown')
        for function in self.domain.functions:
            valid_names.append(function.name)
        util.LOGGER.debug(f"Valid names: {valid_names}")
        # check the states
        util.LOGGER.debug(f"Checking the states")
        for state in self.problem.states:
            result = self.check_state(valid_names, state)
                
        
        util.LOGGER.debug(f"Checking the goals")
        # check the goals
        for name, goals in self.problem.goals.items():
            if name not in valid_names:
                util.LOGGER.debug(f"The agent \"{name}\" is not valid")
                result = False
            for goal in goals:
                goal
                if isinstance(goal, ParsingEpistemicCondition):
                    for agt_name in goal.belief_sequence:
                        if agt_name not in valid_names:
                            util.LOGGER.debug(f"The agent \"{agt_name}\" in {name}'s goal \"{goal}\" is not valid")
                            result = False
                result = self.check_state(valid_names, goal.state)
        
        util.LOGGER.debug(f"Checking the ranges")
        # check the ranges
        for range in self.problem.ranges:
            if range.function_name not in valid_names:
                util.LOGGER.debug(f"The function \"{range.function_name}\" in range is not valid")
                result = False

        util.LOGGER.debug(f"Checking the actions")
        # check the actions
        for action in self.domain.actions:
            for condition in action.pre_conditions:
                variable = condition.state.variable
                target_variable = condition.state.target_variable
                if variable.name not in valid_names or (target_variable.name is not None and target_variable.name not in valid_names):
                    util.LOGGER.debug(f"The action {action.name}'s precondition {condition} is not valid")
                    result = False
        if result:
            util.LOGGER.info(f"The domain and problem pass the checker.")
        else:
            util.LOGGER.info(f"The domain and problem did not pass the checker.")
        return result
        
    def check_state(self, valid_names, state: ParsingState):
        result = True
        variable_entities = state.variable.parameters
        if state.variable.name not in valid_names or (state.target_variable.name is not None and state.target_variable.name not in valid_names):
            util.LOGGER.debug(f"The variable {state.variable.name} is not valid.")
        for entity in variable_entities:
            if entity not in valid_names:
                util.LOGGER.debug(f"The entity \"{entity}\" in state \"{state}\" is not valid")
                result = False
        target_variable_entities = state.target_variable.parameters
        for entity in target_variable_entities:
            if entity not in valid_names:
                util.LOGGER.debug(f"The entity \"{entity}\" in state \"{state}\" is not valid")
                result = False
        return result

def check_goal_conflicts(model: Model):
    if model.problem_type == ProblemType.COOPERATIVE:
        # low_level_goals = get_low_level_goal_set(model.agents[0].goals)
        return check_conflict(model.agents[0].own_goals)
    else:
        for agent in model.agents:
            # low_level_goals = get_low_level_goal_set(agent.goals)
            if not check_conflict(agent.own_goals): return False
        return True

def get_low_level_goal_set(goals: list[Condition]) -> list[Condition]:
    result = []
    for goal in goals:
        result.append(goal)
        low_level_goal = get_low_level_goal(goal)
        if low_level_goal is not None: result.append(low_level_goal)
    return result

def get_low_level_goal(goal: Condition) -> Condition:
    if len(goal.belief_sequence) > 0:
        low_level_goal = Condition()
        low_level_goal.belief_sequence = goal.belief_sequence[1:]
        if len(low_level_goal.belief_sequence) > 0:
            low_level_goal.ep_operator = goal.ep_operator
            low_level_goal.ep_truth = goal.ep_truth
        low_level_goal.condition_operator = goal.condition_operator
        low_level_goal.condition_function_name = goal.condition_function_name
        low_level_goal.condition_function_parameters = goal.condition_function_parameters
        low_level_goal.value = goal.value
        low_level_goal.target_function_name = goal.target_function_name
        low_level_goal.target_function_parameters = goal.target_function_parameters
        return low_level_goal
    else:
        return None

def check_conflict(goals: list[Condition]):
    for i in range(len(goals) - 1):
        for j in range(i + 1, len(goals)):
            goal1 = goals[i]
            goal2 = goals[j]
            if (goal1.belief_sequence == goal2.belief_sequence
                and goal1.condition_function_name == goal2.condition_function_name
                and goal1.condition_function_parameters == goal2.condition_function_parameters):
                # two goals with different ontic value or target function
                if (goal1.ep_operator == goal2.ep_operator
                    and goal1.ep_truth == goal2.ep_truth
                    and goal1.condition_operator == goal2.condition_operator):
                    if goal1.value is not None:
                        if goal1.value != goal2.value:
                            return False
                    else:
                        if (goal1.goal_function_name != goal2.condition_function_name
                            or goal1.goal_function_parameters != goal2.condition_function_parameters):
                            return False
                # two goals with same ontic value or target function, but with different epistemic logics
                elif (goal1.value == goal2.value
                      and goal1.condition_function_name == goal2.condition_function_name
                      and goal1.condition_function_parameters == goal2.condition_function_parameters):
                    # ep operator and truth are the same but condition operator is different
                    if (goal1.ep_operator == goal2.ep_operator
                        and goal1.ep_truth == goal2.ep_truth
                        and goal1.condition_operator != goal2.condition_operator):
                        return False
                    # ep operator and condition operator are the same but truth is different
                    elif (goal1.ep_operator == goal2.ep_operator
                          and goal1.condition_operator == goal2.condition_operator
                          and goal1.ep_truth != goal2.ep_truth):
                        return False
                    # ep truth and condition operator are the same but ep operator is different
                    elif (goal1.ep_truth == goal2.ep_truth
                         and goal1.condition_operator == goal2.condition_operator
                         and goal1.ep_operator != goal2.ep_operator):
                        return False
    return True