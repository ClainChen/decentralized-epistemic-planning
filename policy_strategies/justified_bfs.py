from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action, Function
import heapq
import util
import logging
import random
import copy

LOGGER_LEVEL = logging.DEBUG

class JustifiedBFS(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)

    def get_policy(self, model: Model, agent_name: str) -> Action:
        model_copy = copy.deepcopy(model)
        successors = model_copy.get_agent_successors(agent_name)
        # print([succ.header() for succ in successors])
        if len(successors) > 1:
            possible_successors = [succ.header() for succ in successors]
            samples, expands, virutal_model_num = self.bfs(model_copy, agent_name)
            output = f"Num of virtual models: {virutal_model_num}\n"
            output += f"Num of node expansions: {expands}\n"
            output += f"{[(key, value[1]) for key, value in samples.items()]}\n"
            output += f"{dict([(agent.name, len(agent.all_possible_goals)) for agent in model_copy.agents])}"

            # print(output)
            self.logger.info(f"{output}")
            
            # print(f"{[f'{key} : {value[1]}' for key, value in samples.items()]}")
            samples = {key: value for key, value in samples.items() if key in possible_successors}
            max_value = -1
            for value in samples.values():
                if value[1] > max_value:
                    max_value = value[1]
            if max_value == -1:
                self.logger.info(f"Didn't get result")
                return Action.stay_action(agent_name) if len(successors) == 0 else random.choice(successors)
            possible_actions = [value[0] for _, value in samples.items() if value[1] == max_value]
            return random.choice(possible_actions)
        elif len(successors) == 1:
            self.logger.info(f"Only one successor: {successors[0].header()}")
            return successors[0]
        else:
            stay = Action.stay_action(agent_name)
            self.logger.info(f"No successor, use stay action: {stay.header()}")
            return stay
        
    def bfs(self, model: Model, agent_name: str):
        all_virtual_model = util.generate_virtual_model(model, agent_name)
        samples = {}
        expands = 0
        for virtual_model in all_virtual_model:
            this_sample, this_expand = self.single_bfs(virtual_model, agent_name)
            expands += this_expand
            for key, value in this_sample.items():
                if key in samples:
                    samples[key][1] += value[1]
                else:
                    samples[key] = value
            
            # print("no solution")
        return samples, expands, len(all_virtual_model)

    def single_bfs(self, virtual_model: Model, agent_name: str):
        start_agent = virtual_model.get_agent_by_name(agent_name)
        # self.logger.debug(f"{virtual_model}")
        expand = 1
        samples = {}
        heap: list[util.BFSNode] = []
        current_agent_index = virtual_model.get_agent_index_by_name(agent_name)
        count_agent = len(virtual_model.agents)
        heapq.heappush(heap, util.BFSNode(current_agent_index, [], virtual_model, 0))
        existed_epistemic_world = {agt.name: set() for agt in virtual_model.agents}
        find_solution_depth = -1
        while heap:
            node = heapq.heappop(heap)
            if ((find_solution_depth != -1 and len(node.actions) > find_solution_depth)):
                break

            if node.model.full_goal_complete():
                    find_solution_depth = len(node.actions)
                    if len(node.actions) > 0:
                        string = node.actions[0].header()
                        if string not in samples:
                            samples[string] = [node.actions[0], 1]
                        else:
                            samples[string][1] += 1
                    self.logger.debug(f"Complete path: {[action.header() for action in node.actions]}")
                    continue
            current_agent = node.model.agents[node.current_index]
            # 检查当前世界状态中，对于agent_name代理来说是否有笃定current_agent会做的行为
            # 如果有，则直接讲这些行为记为successors，如果没有则正常生成successors
            jp_world_for_agent_name = [f.id for f in util.get_epistemic_world(node.model, [agent_name])]
            hash_set_jp_world = frozenset(jp_world_for_agent_name)
            successors = list(start_agent.E[hash_set_jp_world][current_agent.name])
            successors = [succ for succ in successors if util.is_valid_action(node.model, succ)]
            # if current_agent.name == agent_name:
            #     poss_succs = node.model.get_agent_successors(current_agent.name)
            #     successors = [succ for succ in poss_succs if succ not in successors]
            if len(successors) == 0:
                successors = node.model.get_agent_successors(current_agent.name)
            
            for succ in successors:
                next_model = node.model.copy()
                next_model.move(current_agent.name, succ)
                # 过滤机制
                observe_funcs = frozenset([frozenset([agt.name] + [f.id for f in util.get_epistemic_world(next_model, [agt.name])]) for agt in next_model.agents])
                if observe_funcs in existed_epistemic_world[current_agent.name]:
                    # virtual_model.logger.debug(f"Pruned path: {[action.header() for action in node.actions] + [succ.header()]}")
                    continue
                existed_epistemic_world[current_agent.name].add(observe_funcs)

                heapq.heappush(heap, 
                            util.BFSNode((node.current_index + 1) % count_agent,
                                        node.actions + [succ],
                                        next_model,
                                        node.priority + 1))
                expand += 1
        return samples, expand