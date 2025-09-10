from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action, Function
import heapq
import util
import logging
import random
import copy
import time

LOGGER_LEVEL = logging.DEBUG

class SeqJustifiedBFS(AbstractPolicyStrategy):
    """
    The basic idea is still justified BFS, but instead of locally simulate in round based, Sequence Justified BFS will only let the first expansion be the action of given agent_name, and the remain actions are not round based, which means any agents can move in the next step.
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
        # print(f"Generated {len(all_virtual_model)} virtual models")
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
        heapq.heappush(heap, util.BFSNode(0, [], virtual_model, 0))
        existed_epistemic_world = set()
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
                    # util.LOGGER.debug(f"Complete path: {[action.header() for action in node.actions]}")
                    continue
            if node.current_index == 0:
                current_agent = [agent_name]
            else:
                current_agent = [agt.name for agt in virtual_model.agents]
            # 检查当前世界状态中，对于agent_name代理来说是否有笃定current_agent会做的行为
            # 如果有，则直接讲这些行为记为successors，如果没有则正常生成successors
            jp_world_for_agent_name = [f.id for f in util.get_epistemic_world(node.model, [agent_name])]
            hash_set_jp_world = frozenset(jp_world_for_agent_name)
            successors = {ca: start_agent.get_E(hash_set_jp_world, ca) for ca in current_agent}
            successors = {key: [succ for succ in value if util.is_valid_action(node.model, succ)] 
                          for key, value in successors.items()}
            # if current_agent.name == agent_name:
            #     poss_succs = node.model.get_agent_successors(current_agent.name)
            #     successors = [succ for succ in poss_succs if succ not in successors]
            for key, value in successors.items():
                if len(value) == 0:
                    successors[key] = node.model.get_agent_successors(key)
            
            for name, succs in successors.items():
                for succ in succs:
                    next_model = node.model.copy()
                    next_model.move(name, succ)
                    # 过滤机制
                    observe_funcs = frozenset([frozenset([agt.name] + [f.id for f in util.get_epistemic_world(next_model, [agt.name])]) for agt in next_model.agents])
                    if observe_funcs in existed_epistemic_world:
                        continue
                    existed_epistemic_world.add(observe_funcs)

                    heapq.heappush(heap, 
                                util.BFSNode(-1,
                                            node.actions + [succ],
                                            next_model,
                                            node.priority + 1))
                    expand += 1
        return samples, expand