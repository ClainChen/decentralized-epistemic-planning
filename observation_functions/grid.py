import logging
from abstracts import AbstractObservationFunction


LOGGER_LEVEL = logging.DEBUG

class GridObsFunc(AbstractObservationFunction):

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
            - what in its own location
            - the survivor locations that have been shared to it
            - other agent's location if they are in sharing state
        """
        result = set()
        sharing_agent = ''
        survivor_loc_shared = {}
        survivor_loc = {}
        agent_loc = {}
        searched = {}
        receivable = {}

        for func in functions:
            if func.name == 'agent_loc':
                agent_loc[func.parameters['?a']] = func
            elif func.name == 'shared' and func.value == 1:
                survivor_loc_shared[func.parameters['?s']] = func
            elif func.name == 'survivor_loc':
                survivor_loc[func.parameters['?s']] = func
            elif func.name == 'searched':
                searched[func.parameters['?l']] = func
            else:
                result.add(func)
                if func.name == 'receivable':
                    receivable[func.parameters['?a']] = func.value
                elif func.name == 'sharing' and func.value == 1:
                    sharing_agent = func.parameters['?a']

        # if this agent's location is not in the function (in nesting observation), that means the last agent cannot observe this agent, then it should have only common knowledge
        if agent_name not in agent_loc:
            return list(result)
        
        # agent location
        for loc_func in agent_loc.values():
            if loc_func.value == agent_loc[agent_name].value:
                result.add(loc_func)
        # searched status of its own location
        result.add(searched[agent_loc[agent_name].value])
        # if this agent is sharing, then it knows the survivor in this location is shared
        for s, loc_func in survivor_loc.items():
            if loc_func.value == agent_loc[agent_name].value:
                result.add(loc_func)
                if sharing_agent == agent_name and s in survivor_loc_shared:
                    result.add(survivor_loc_shared[s])

        # if this agent is not receivable, then it will only observe what in its own location
        if receivable[agent_name] == 0:
            return list(result)
        
        # if this agent is receivable, then it will also observe other sharing agent's location information, and the location of the shared survivors.
        if sharing_agent in agent_loc and sharing_agent != agent_name and sharing_agent != '':
            result.add(agent_loc[sharing_agent])
            # searched status of the sharing agent's location
            if agent_loc[sharing_agent].value in searched:
                result.add(searched[agent_loc[sharing_agent].value])
        for s, f in survivor_loc_shared.items():
            result.add(f)
            if f.value == 1 and s in survivor_loc:
                result.add(survivor_loc[s])

        return list(result)
        
    
    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents
        
        