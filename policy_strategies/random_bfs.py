from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action, Function
import heapq
import util
import logging
from collections import Counter
import random

LOGGER_LEVEL = logging.DEBUG

class RandomBFS(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        if len(successors) > 1:
            return self.bfs(model, agent_name)
        elif len(successors) == 1:
            return successors[0]
        else:
            return None
        
    def bfs(self, model: Model, agent_name: str):
        # observed_world = []
        heap: list[BFSNode] = []
        # self.logger.debug(f"current model: {model}")
        virtual_model = random.choice(util.generate_virtual_model(model, agent_name))
        # self.logger.debug(f"virtual model\n{virtual_model}")
        # exit(0)
        # virtual_model = model.copy()
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
                    return node.actions[0] if node.actions else None
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
        return None

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
