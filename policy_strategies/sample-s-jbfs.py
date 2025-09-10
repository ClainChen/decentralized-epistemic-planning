from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action, Function
import heapq
import util
import logging
import random
import math

class SampleSeqJustifiedBFS(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """

    def get_policy(self, model: Model, agent_name: str) -> Action:
        model_copy = model.copy()
        successors = model_copy.get_agent_successors(agent_name)
        if len(successors) > 1:
            samples = self.random_pick_bfs(model_copy, agent_name)
            if len(samples) == 0:
                return Action.stay_action(agent_name)
            succs = [value for value in samples.values()]
            succs.sort(reverse=True, key=lambda x: x[1])
            maxx = succs[0][1]
            succs = [value[0] for value in succs if value[1] == maxx]
            return random.choice(succs)
        elif len(successors) == 1:
            return successors[0]
        else:
            return Action.stay_action(agent_name)
        
    def random_pick_bfs(self, model: Model, agent_name: str):
        vms = util.generate_virtual_model(model, agent_name)
        vms = random.choices(vms, k=max(1, math.ceil(len(vms) / 2)))
        samples = {}
        for vm in vms:
            sample = self.single_bfs(vm, agent_name)
            for key, value in sample.items():
                if key not in samples:
                    samples[key] = [value[0], 0]
                samples[key][1] += value[1]
        return samples


    def single_bfs(self, virtual_model: Model, agent_name: str):
        start_agent = virtual_model.get_agent_by_name(agent_name)
        # util.LOGGER.debug(f"{virtual_model}")
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
        return samples
