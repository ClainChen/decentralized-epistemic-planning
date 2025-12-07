from abstracts import AbstractPolicyStrategy
from dep.epistemic_class import Model, Action
import heapq
import util
import logging
import random
import copy
import time
import cache_helper as ch

LOGGER_LEVEL = logging.DEBUG


class ShareGoalBFS(AbstractPolicyStrategy):
    """
    1. Agent will consider all possible goals
    2. Agent will not consider the experience
    """

    def get_policy(self, model: Model, agent_name: str) -> tuple[Action, int]:
        model_copy = copy.deepcopy(model)
        successors = model_copy.get_agent_successors(agent_name)
        nv = 0
        # print(f"{agent_name}: {[succ.header() for succ in successors]}")
        if len(successors) > 1:
            possible_successors = [succ.header() for succ in successors]
            samples, num_vms = self.bfs(model_copy, agent_name)
            nv += num_vms
            output = f"{[(key, value[1]) for key, value in samples.items()]}\n"
            output += f"{dict([(agent.name, len(agent.all_possible_goals)) for agent in model_copy.agents])}"
            util.LOGGER.info(f"{output}")

            succs = [value for value in samples.values() if value[0].header() in possible_successors]
            if len(succs) == 0:
                return Action.stay_action(agent_name), nv
            succs.sort(reverse=True, key=lambda x: x[1])
            maxx = succs[0][1]
            succs = [value[0] for value in succs if value[1] == maxx]
            return random.choice(succs), nv
        elif len(successors) == 1:
            util.LOGGER.info(f"Only one successor: {successors[0].header()}")
            return successors[0], 0
        else:
            stay = Action.stay_action(agent_name)
            util.LOGGER.info(f"No successor, use stay action: {stay.header()}")
            return stay, 0

    def bfs(self, model: Model, agent_name: str):
        all_virtual_model = util.generate_virtual_model(model, agent_name)
        num_vms = len(all_virtual_model)
        print(f"{agent_name} vms: {len(all_virtual_model)}")
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
        util.LOGGER.info(
            f"Models: {len(all_virtual_model)}, Exapnds: {expands}, {(((time.perf_counter() - start) / expands) * 1e3):.3f}ms/expand")
        return samples, num_vms

    def single_bfs(self, virtual_model: Model, agent_name: str):
        expand = 1
        samples = {}
        heap: list[util.BFSNode] = []
        heapq.heappush(heap, util.BFSNode(0, [], [virtual_model], []))
        existed_epistemic_world = set()
        find_solution_depth = -1
        while heap:
            node = heapq.heappop(heap)
            cur_model = node.model[-1]

            # this is the cache pruning mechanism, just a try
            l = ch.BFS_CACHE.get_cache(cur_model)
            # print(l, find_solution_depth)

            # if agent_name == 'a':
            #     print([act.header() for act in node.actions])
            if len(node.actions) > 0 and node.actions[0].header() in samples:
                continue

            if find_solution_depth != -1 and len(node.actions) > find_solution_depth:
                break

            # if cur_model.full_goal_complete():
            if cur_model.full_goal_complete() or (len(node.model) > 1 and l
                                                  and (find_solution_depth == -1
                                                       or len(node.actions) + l <= find_solution_depth)):

                if l:
                    # print(l)
                    find_solution_depth = len(node.actions) + l
                    ch.BFS_CACHE.add_cache(node, l)
                else:
                    find_solution_depth = len(node.actions)
                    ch.BFS_CACHE.add_cache(node)

                # find_solution_depth = len(node.actions)
                samples = {k: v for k, v in samples.items() if v[1] <= find_solution_depth}
                if len(node.actions) > 0:
                    action = node.actions[0]
                    string = action.header()
                    if string not in samples:
                        samples[string] = [action, find_solution_depth]
                    # else:
                    #     samples[string][1] += 1
                # util.LOGGER.debug(f"Complete path: {[action.header() for action in node.actions]}")
                continue
            if node.current_index == 0:
                current_agent = [agent_name]
            else:
                current_agent = [agt.name for agt in virtual_model.agents]
            successors = {ca: cur_model.get_agent_successors(ca) for ca in current_agent}
            for name, succs in successors.items():
                for succ in succs:
                    if 'stay' in succ.name and len(node.actions) > 0:
                        continue
                    next_model = cur_model.copy()
                    next_model.move(name, succ)
                    observe_funcs = tuple(
                        tuple([''.join(bs)] + [s.id for s in util.get_epistemic_world(next_model, bs)])
                        for bs in next_model.possible_belief_sequences)
                    observe_funcs = frozenset(observe_funcs)
                    if observe_funcs in existed_epistemic_world:
                        # print('here')
                        continue
                    existed_epistemic_world.add(observe_funcs)

                    heapq.heappush(heap,
                                   util.BFSNode(-1,
                                                node.actions + [succ],
                                                node.model + [next_model],
                                                node.agents + [name]))
                    expand += 1
        # if len(samples) == 0:
        #     print(virtual_model)
        #     exit(0)
        return {k: [v[0], 1] for k, v in samples.items()}, expand
