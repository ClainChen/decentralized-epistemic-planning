from util import QuickQueryFunctions
import logging
from dep.epistemic_class import Model, Function
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG


class Deliver2rObsFunc(AbstractObservationFunction):
    def get_observable_functions(self, functions: list[Function], agent_name: str, all_funcs, ontic_functions) -> list[Function]:
        ff: QuickQueryFunctions = QuickQueryFunctions.build_qqf(functions)
        result = set()
        current_agent_loc = ff.get('agent_loc', {'?a': agent_name})
        all_item_in_same_loc = all(f.value == current_agent_loc for f in ff.get_by_name('item_loc'))

        for func in functions:
            if func.name == 'hold':
                """
                hold: '?a' = item
                1. if the agent in the same place as this agent, then agent know it
                2. if all item in the same loc as this agent, then agent know it
                """
                agt = func.parameters['?a']
                if all_item_in_same_loc or current_agent_loc == ff.get('agent_loc', {'?a': agt}):
                    result.add(func)
            elif func.name == 'is_free':
                """
                1. if the agent can see the item, then agent know it
                """
                item = func.parameters['?i']
                if current_agent_loc == ff.get('item_loc', {'?i': item}) or item == 'nothing':
                    result.add(func)
            else:
                result.add(func)

        return list(result)

    def get_observable_agents(self, model, functions, agent_name):
        agent_room = {}
        for func in functions:
            if func.name == 'agent_loc':
                agent_room[func.parameters['?a']] = func.value
        current_agent_room = agent_room[agent_name]
        return [agent for agent, room in agent_room.items() if room == current_agent_room]

    def post_process_jp(self, functions, agent_name, all_funcs, ontic_functions):
        return functions[:]
