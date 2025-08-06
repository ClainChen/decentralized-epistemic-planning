from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action, Function
import heapq
import util
import logging
from collections import Counter
import random
import threading

LOGGER_LEVEL = logging.DEBUG

class CompleteBFS(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        successors = [succ for succ in successors if util.is_valid_action(model, succ, agent_name)]
        if len(successors) > 1:
            samples, expands = self.bfs(model, agent_name)
            # print(f"{[f'{key} : {value[1]}' for key, value in samples.items()]}")
            max_value = max([v[1] for v in samples.values()])
            possible_actions = [value[0] for _, value in samples.items() if value[1] == max_value]
            print(f"Num of node expansions: {expands}")
            return random.choice(possible_actions)
        elif len(successors) == 1:
            return successors[0]
        else:
            return None
        
    def bfs(self, model: Model, agent_name: str):
        all_virtual_model = util.generate_virtual_model(model, agent_name)
        # print(f'Virutal Models Num: {len(all_virtual_model)}')
        # self.logger.debug(f"Show virtual models: {len(all_virtual_model)}")
        # for virtual_model in all_virtual_model:
        #     self.logger.debug(f"{virtual_model}")
        # exit(0)
        # if len(all_virtual_model) == 0:
        #     all_virtual_model = [model]
        samples = {}
        expands = [0]
        lock = threading.Lock()
        threads = []
        for virtual_model in all_virtual_model:
            t = threading.Thread(target=self.single_bfs, 
                                 args=(samples, virtual_model, agent_name, expands, lock))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
            
            # print("no solution")
        return samples, expands[0]

    def single_bfs(self, samples, virtual_model: Model, agent_name: str, expands, lock):
        # self.logger.debug(f"{virtual_model}")
        expand = 1
        heap: list[BFSNode] = []
        current_agent_index = virtual_model.get_agent_index_by_name(agent_name)
        count_agent = len(virtual_model.agents)
        heapq.heappush(heap, BFSNode(current_agent_index, [], virtual_model, 0))
        # observed_world.append(virtual_model.ontic_functions)
        while heap:
            node = heapq.heappop(heap)
            if node.priority == 12:
                # print("no solution")
                break
            if node.model.full_goal_complete():
                    # print(f"{[f"{action.name}({list(action.parameters.values())})" for action innode.actions]}")
                    # self.logger.debug(f"{node.model}")
                    string = node.actions[0].header()
                    with lock:
                        expands[0] += expand
                        if string not in samples:
                            samples[string] = [node.actions[0], 1]
                        else:
                            samples[string][1] += 1
                    # print({f'{value[0].name}({list(value[0].parameters.values())})': value[1] for key, value in samples.items()})
                    break
            current_agent = node.model.agents[node.current_index]
            successors = node.model.get_agent_successors(current_agent.name)
            successors = [succ for succ in successors if util.is_valid_action(node.model, succ, current_agent.name)]
            # result = "start function\n"
            # if len(node.model.history_functions) > 0:
            #     for f in node.model.history_functions[0]:
            #         result += f"{f}\n"
            # result += "current function\n"
            # for f in node.model.ontic_functions:
            #     result += f"{f}\n"
            # result += f"action sequence:\n"
            # for action in node.actions:
            #     result += f"{action.header()}\n"
            # result += f"current successors:\n"
            # for succ in successors:
            #     result += f"{succ.header()}\n"
            # self.logger.debug(f"{result}")

            for succ in successors:
                expand += 1
                if (node.actions and succ.name == "stay"
                    and all([action.name == 'stay' for action in node.actions[-count_agent:]])):
                    continue
                next_model = node.model.copy()
                next_model.move(current_agent.name, succ)
                heapq.heappush(heap, 
                            BFSNode((node.current_index + 1) % count_agent,
                                        node.actions + [succ],
                                        next_model,
                                        node.priority + 1))

class BFSNode:
    def __init__(self, current_index, action, model, priority):
        self.current_index: int = current_index
        self.actions: list[Action] = action
        self.model: Model = model
        self.priority: int = priority
    
    def __lt__(self, other):
        return self.priority < other.priority
