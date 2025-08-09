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
            possible_successors = [succ.header() for succ in successors]
            samples, expands = self.bfs(model, agent_name)
            print(f"Num of node expansions: {expands}")
            # print(f"{[f'{key} : {value[1]}' for key, value in samples.items()]}")
            samples = {key: value for key, value in samples.items() if key in possible_successors}
            max_value = -1
            for value in samples.values():
                if value[1] > max_value:
                    max_value = value[1]
            if max_value == -1:
                return Action.stay_action(agent_name)
            possible_actions = [value[0] for _, value in samples.items() if value[1] == max_value]
            return random.choice(possible_actions)
        elif len(successors) == 1:
            return successors[0]
        else:
            return Action.stay_action(agent_name)
        
    def bfs(self, model: Model, agent_name: str):
        all_virtual_model = util.generate_virtual_model(model, agent_name)
        print(f'Virutal Models Num: {len(all_virtual_model)}')
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
        heap: list[util.BFSNode] = []
        current_agent_index = virtual_model.get_agent_index_by_name(agent_name)
        count_agent = len(virtual_model.agents)
        heapq.heappush(heap, util.BFSNode(current_agent_index, [], virtual_model, 0))
        existed_epistemic_world = {agt.name: set() for agt in virtual_model.agents}
        # observed_world.append(virtual_model.ontic_functions)
        find_solution_depth = -1
        while heap:
            node = heapq.heappop(heap)
            # if node.priority == 12:
            #     # print("no solution")
            #     break
            if find_solution_depth != -1 and len(node.actions) > find_solution_depth:
                break

            if node.model.full_goal_complete():
                    # print(f"{[f"{action.name}({list(action.parameters.values())})" for action innode.actions]}")
                    # self.logger.debug(f"{node.model}")
                    find_solution_depth = len(node.actions)
                    if len(node.actions) > 0:
                        string = node.actions[0].header()
                        with lock:
                            if string not in samples:
                                samples[string] = [node.actions[0], 1]
                            else:
                                samples[string][1] += 1
                        # print({f'{value[0].name}({list(value[0].parameters.values())})': value[1] for key, value in samples.items()})
                    continue
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
                next_model = node.model.copy()
                next_model.move(current_agent.name, succ)
                # 过滤机制
                ep_funcs = []
                for agt in next_model.agents:
                    his_ep_funcs = next_model.get_history_functions_of_agent(agt.name)
                    cur_ep_funcs = next_model.get_functions_of_agent(agt.name)
                    ep_funcs.append(frozenset(util.get_epistemic_world(reversed(his_ep_funcs + [cur_ep_funcs]))))
                if frozenset(ep_funcs) in existed_epistemic_world[current_agent.name]:
                    continue
                existed_epistemic_world[current_agent.name].add(frozenset(ep_funcs))

                heapq.heappush(heap, 
                            util.BFSNode((node.current_index + 1) % count_agent,
                                        node.actions + [succ],
                                        next_model,
                                        node.priority + 1))
                expand += 1
        with lock:
            expands[0] += expand