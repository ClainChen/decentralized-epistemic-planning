import logging

import util
from abstracts import AbstractObservationFunction
from util import QuickQueryFunctions

LOGGER_LEVEL = logging.DEBUG


class GridObsFunc(AbstractObservationFunction):
    # agent_searched = {}

    def get_observable_functions(self, functions, agent_name, all_funcs, ontic_functions):
        """
        1. if an agent is not receivable, it can only observe what in its own location and the common knowledge.
        2. if an agent is receivable, it can observe:
            - common knowledge
                - movable
                - sharable
                - receivable
                - loc_id
                - connected
                - sharing
                - share_lock
                - agent_type
            - what in its own location
            - the survivor locations that have been shared to it
            - other agent's location if they are in sharing state
        """
        if 'qh' in agent_name:
            return functions[:]

        ff: util.QuickQueryFunctions = QuickQueryFunctions.build_qqf(functions)
        result = set()

        current_agent_loc = ff.get('agent_loc', {'?a': agent_name})
        receivable = ff.get('receivable', {'?a': agent_name}) == 1
        sharing = [ff.get('agent_loc', {'?a': f.parameters['?a']})
                   for f in ff.get_by_name('sharing') if f.value == 1]

        for func in functions:
            if func.name == 'agent_loc':
                """
                1. assume agent will always knows the agents that not movable
                2. agent will always knows the special quieter agents
                3. agent can observe the other agents in the same location
                4. receivable agent can observe the other agents if they are sharing
                """
                agt = func.parameters['?a']
                if (ff.get('movable', {'?a': agt}) == 0  #1
                        or 'qh' in agt  #2
                        or current_agent_loc == func.value  #3
                        or (receivable and ff.get('sharing', {'?a': agt}) == 1)):  #4
                    result.add(func)
            elif func.name == 'survivor_loc':
                """
                1. if the survivor is being shared, receivable agent will know it
                2. if the survivor in the same location, agent will know it
                """
                sur = func.parameters['?s']
                if ((receivable and ff.get('shared', {'?s': sur}) == 1)
                        or ff.get('survivor_loc', {'?s': sur}) == current_agent_loc):
                    result.add(func)
            elif func.name == 'shared':
                """
                1. if an survivor is being shared, receivable agent will know it
                2. if this agent is currently sharing and in the same location with the this survivor, agent will know it
                """
                sur = func.parameters['?s']
                if ((receivable and func.value == 1)
                        or (ff.get('sharing', {'?a': agent_name})
                            and ff.get('survivor_loc', {'?s': sur}) == current_agent_loc)):
                    result.add(func)
            elif func.name == 'searched':
                """
                1. if this agent is receivable, then this agent will know it
                2. if this agent in this location, then this agent will know it
                """
                loc = func.parameters['?l']
                if current_agent_loc == loc or (receivable and loc in sharing):
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
        return [
            all_funcs.get_function(f.name, f.parameters, 0)
            if (f.name == 'searched' and f.value != 1) or (f.name == 'shared' and f.value != 1)
            else f
            for f in functions
        ]
