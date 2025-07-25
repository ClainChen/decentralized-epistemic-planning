from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action, Function
import heapq
import util
import logging
from collections import Counter
import random

LOGGER_LEVEL = logging.DEBUG

class CompleteBFS(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        if len(successors) > 1:
            samples = self.bfs(model, agent_name)
            min_value = min(samples.values())
            possible_actions = [action for action, value in samples.items() if value == min_value]
            return random.choice(possible_actions)
        elif len(successors) == 1:
            return successors[0]
        else:
            return None
        
    def bfs(self, model: Model, agent_name: str):
        all_virtual_model = util.generate_virtual_model(model, agent_name)
        samples = {}

        for virtual_model in all_virtual_model:
            heap: list[BFSNode] = []
            current_agent_index = virtual_model.get_agent_index_by_name(agent_name)
            count_agent = len(virtual_model.agents)
            heapq.heappush(heap, BFSNode(current_agent_index, [], virtual_model, 0))
            # observed_world.append(virtual_model.ontic_functions)
            while heap:
                node = heapq.heappop(heap)
                if node.priority == 10:
                    break
                if node.model.full_goal_complete():
                        # print(f"{[f"{action.name}({list(action.parameters.values())})" for action in node.actions]}")
                        # self.logger.debug(f"{node.model}")
                        if node.actions[0] not in samples:
                            samples[node.actions[0]] = 1
                        else:
                            samples[node.actions[0]] += 1
                        break
                current_agent = node.model.agents[node.current_index]
                successors = node.model.get_agent_successors(current_agent.name)
                successors = [succ for succ in successors if util.is_valid_action(node.model.ontic_functions, succ)]
                # print(f"{current_agent.name}: {[f"{action.name}({action.parameters})" for action in node.actions]} {len(node.actions)}")


                for succ in successors:
                    if node.actions and succ.name == "stay" and node.actions[-1].name == "stay":
                        continue
                    next_model = node.model.copy()
                    next_model.do_action(current_agent.name, succ)
                    for agent in next_model.agents:
                        next_model.observe_and_update_agent(agent.name)
                    heapq.heappush(heap, 
                                BFSNode((node.current_index + 1) % count_agent,
                                            node.actions + [succ],
                                            next_model,
                                            node.priority + 1))
            # print("no solution")
        return samples

class BFSNode:
    def __init__(self, current_index, action, model, priority):
        self.current_index: int = current_index
        self.actions: list[Action] = action
        self.model: Model = model
        self.priority: int = priority
    
    def __lt__(self, other):
        return self.priority < other.priority

def world_observed(worlds, functions):
    return any([Counter(functions) == Counter(w) for w in worlds])
