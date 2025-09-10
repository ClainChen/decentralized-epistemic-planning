from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action, Function
import heapq
import util
import time
import logging
import random
import copy

LOGGER_LEVEL = logging.DEBUG

class CompleteBFS(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """

    def get_policy(self, model: Model, agent_name: str) -> Action:
        model_copy = copy.deepcopy(model)
        successors = model_copy.get_agent_successors(agent_name)
        # print([succ.header() for succ in successors])
        if len(successors) > 1:
            possible_successors = [succ.header() for succ in successors]
            samples = self.bfs(model_copy, agent_name)
            output = f"{[(key, value[1]) for key, value in samples.items()]}\n"
            output += f"{dict([(agent.name, len(agent.all_possible_goals)) for agent in model_copy.agents])}"
            util.LOGGER.info(f"{output}")
            
            succs = [value for value in samples.values() if value[0].header() in possible_successors]
            if len(succs) == 0:
                return Action.stay_action(agent_name) if len(successors) == 0 else random.choice(successors)
            succs.sort(reverse=True, key=lambda x: x[1])
            maxx = succs[0][1]
            succs = [value[0] for value in succs if value[1] == maxx]
            return random.choice(succs)
        elif len(successors) == 1:
            util.LOGGER.info(f"Only one successor: {successors[0].header()}")
            return successors[0]
        else:
            stay = Action.stay_action(agent_name)
            util.LOGGER.info(f"No successor, use stay action: {stay.header()}")
            return stay
        
    def bfs(self, model: Model, agent_name: str):
        all_virtual_model = util.generate_virtual_model(model, agent_name)
        samples = {}
        expands = 0
        start = time.perf_counter()
        for virtual_model in all_virtual_model:
            this_sample, this_expand = self.single_bfs(virtual_model, agent_name)
            expands += this_expand
            for key, value in this_sample.items():
                if key in samples:
                    samples[key][1] += value[1]
                else:
                    samples[key] = value
        util.LOGGER.info(f"Models: {len(all_virtual_model)}, Exapnds: {expands}, {(((time.perf_counter() - start) / expands) * 1e3):.3f}ms/expand")
            # print("no solution")
        return samples

    def single_bfs(self, virtual_model: Model, agent_name: str):
        start_agent = virtual_model.get_agent_by_name(agent_name)
        # util.LOGGER.debug(f"{virtual_model}")
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
                    util.LOGGER.debug(f"Complete path: {[action.header() for action in node.actions]}")
                    continue
            current_agent = node.model.agents[node.current_index]
            successors = node.model.get_agent_successors(current_agent.name)
            for succ in successors:
                next_model = node.model.copy()
                next_model.move(current_agent.name, succ)
                # 过滤机制
                observe_funcs = frozenset([frozenset([agt.name] + [f.id for f in util.get_epistemic_world(next_model, [agt.name])]) for agt in next_model.agents])
                if observe_funcs in existed_epistemic_world[current_agent.name]:
                    # util.LOGGER.debug(f"Pruned path: {[action.header() for action in node.actions] + [succ.header()]}")
                    continue
                existed_epistemic_world[current_agent.name].add(observe_funcs)

                heapq.heappush(heap, 
                            util.BFSNode((node.current_index + 1) % count_agent,
                                        node.actions + [succ],
                                        next_model,
                                        node.priority + 1))
                expand += 1
        return samples, expand