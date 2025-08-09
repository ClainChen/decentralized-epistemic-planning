from epistemic_handler.epistemic_class import *
import logging
import itertools
import util
import time
import random
from tqdm import tqdm

PROBLEM_BUILDER_LOG_LEVEL = logging.DEBUG

class ProblemBuilder:
    def __init__(self, base_model, handlers, log_level=PROBLEM_BUILDER_LOG_LEVEL):
        self.logger = util.setup_logger(__name__, handlers, logger_level=log_level)
        self.base_model: Model = base_model
    
    def get_all_init_ontic_world(self):
        groups = []
        for schema in self.base_model.function_schemas:
            if isinstance(schema.range, tuple):
                start, end = schema.range
                value_range = list(range(start, end + 1))
            else:
                value_range = schema.range
            
            params = [self.base_model.get_all_entity_name_by_type(type) for type in schema.require_parameters.values()]
            # start grouping
            for param in itertools.product(*params):
                group = []
                for value in value_range:
                    function = Function()
                    function.name = schema.name
                    function.range = schema.range
                    function.type = schema.type
                    function.value = value
                    function.parameters = dict(zip(schema.require_parameters.keys(), param))
                    group.append(function)
                groups.append(group)
            
        # generate all possible function groups
        all_possible_init = [list(lst) for lst in itertools.product(*groups)]
        all_possible_init = [world for world in all_possible_init if self.base_model.rules.check_functions(world)]
        return all_possible_init

    def get_all_poss_goals(self, agent_name = "") -> list[dict[str, list[Condition]]]:
        """
        if input agent_name, then the generated goals for agent_name will only have it's own goals
        """
        
        # Generate all possible belief sequences
        belief_sequences = self.base_model.possible_belief_sequences

        groups = []
        goal_schemas = [schema for schema in self.base_model.function_schemas if schema.name in self.base_model.acceptable_goal_set]
        for schema in goal_schemas:
            if isinstance(schema.range, tuple):
                start, end = schema.range
                value_range = list(range(start, end + 1))
            else:
                value_range = schema.range
            
            params = [self.base_model.get_all_entity_name_by_type(type) for type in schema.require_parameters.values()]
            for sequence in belief_sequences:
                for param in itertools.product(*params):
                    group = []
                    for value in value_range:
                        goal = Condition()
                        goal.belief_sequence = sequence
                        goal.ep_operator = EpistemicOperator.EQUAL
                        goal.ep_truth = EpistemicTruth.TRUE
                        goal.condition_operator = ConditionOperator.EQUAL
                        goal.condition_function_name = schema.name
                        goal.condition_function_parameters = dict(zip(schema.require_parameters.keys(), param))
                        goal.value = value
                        group.append(goal)
                    groups.append(group)
        possible_goals = {agent: [group for group in groups if group[0].belief_sequence[0] == agent] for agent in self.base_model.get_all_entity_name_by_type('agent') if agent != agent_name}
        
        for agent, groups in possible_goals.items():
            if agent == agent_name:
                continue
            goal_sets = []
            for i in range(1, len(groups) + 1):
                pair_combs = itertools.combinations(groups, i)
                for pair_comb in pair_combs:
                    goal_sets += [list(comb) for comb in itertools.product(*pair_comb)]
            possible_goals[agent] = goal_sets
        if agent_name != "":
            possible_goals[agent_name] = [self.base_model.get_agent_by_name(agent_name).own_goals]
        
        combs = [list(comb) for comb in itertools.product(*possible_goals.values())]
        results = []
        agent_goal_sets = [dict(zip(possible_goals.keys(), comb)) for comb in combs]
        agent_goal_sets.sort(key=lambda x: sum([len(value) for value in x.values()]))
        valid = 0
        invalid_jump = 0
        total = len(combs)

        invalid_goal_sets: list[set] = []
        
        print(f"共{len(agent_goal_sets)}组设置，开始测试所有可能的Goals设置")
        with tqdm(range(total), desc="审查进度") as pbar:
            for i in pbar:
                agent_goal_set = agent_goal_sets[i]
                goal_set = set()
                for goals in agent_goal_set.values():
                    for goal in goals:
                        goal_set.add(goal)
                jump= False
                for sett in invalid_goal_sets:
                    if sett.issubset(goal_set): 
                        jump = True
                        invalid_jump += 1
                        break
                if not jump:          
                    test_model = self.base_model.copy()
                    for agent in test_model.agents:
                        agent.own_goals = agent_goal_set[agent.name]
                    is_valid = util.check_bfs(test_model)
                    if is_valid:
                        results.append(agent_goal_set)
                        valid += 1
                    else:
                        invalid_goal_sets.append(goal_set)
                pbar.set_postfix({"有效数": f"{valid}/{total}", "跳过无效审查数": invalid_jump})
        
        # for sett in results:
        #     result = "\n"
        #     for agent, goals in sett.items():
        #         result += f"{agent}:\n"
        #         for goal in goals:
        #             result += f"{goal}\n"
        #         result += f"-----\n"
        #     self.logger.debug(result)
        
        return results

    def get_all_poss_problem(self, worlds: list[list[Function]], goals: dict[str, list[list[Condition]]]):
        # The difference between the problems is that their initial states and agent's goals are different
        # so that are what we needs to change.

        # filter the acceptable goals first
        # now assume if a goal is valid in a world settings, then it is valid in all world settings.
        
        pass





