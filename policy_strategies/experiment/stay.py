from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action, Function
import heapq
import util
import logging
import random
import copy
import time

LOGGER_LEVEL = logging.DEBUG

class StayBFS(AbstractPolicyStrategy):
    """
    1. Only the given agent will move
    2. Run in sequential order
    3. Finish expand when the given agent's goal is complete
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
        return samples

    def single_bfs(self, virtual_model: Model, agent_name: str):
        expand = 1
        samples = {}
        heap: list[util.BFSNode] = []
        heapq.heappush(heap, util.BFSNode(0, [], virtual_model))
        existed_epistemic_world = set()
        find_solution_depth = -1
        while heap:
            node = heapq.heappop(heap)
            if ((find_solution_depth != -1 and len(node.actions) > find_solution_depth)):
                break

            if node.model.agent_goal_complete(agent_name):
                    find_solution_depth = len(node.actions)
                    action = node.actions[0] if len(node.actions) > 0 else Action.stay_action(agent_name)
                    string = action.header()
                    if string not in samples:
                        samples[string] = [action, 1]
                    else:
                        samples[string][1] += 1
                    # util.LOGGER.debug(f"Complete path: {[action.header() for action in node.actions]}")
                    continue
            successors = {agent_name: node.model.get_agent_successors(agent_name)}
            for name, succs in successors.items():
                for succ in succs:
                    next_model = node.model.copy()
                    next_model.move(name, succ)

                    observe_funcs = frozenset([frozenset([agt.name] + [f.id for f in util.get_epistemic_world(next_model, [agt.name])]) for agt in next_model.agents])
                    if observe_funcs in existed_epistemic_world:
                        continue
                    existed_epistemic_world.add(observe_funcs)

                    heapq.heappush(heap, 
                                util.BFSNode(-1,
                                            node.actions + [succ],
                                            next_model))
                    expand += 1
        return samples, expand