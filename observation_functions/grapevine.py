import util
import logging
from abstracts import AbstractObservationFunction
import copy

LOGGER_LEVEL = logging.DEBUG

class GrapevineObsFunc(AbstractObservationFunction):
    
    def get_observable_functions(self, model, functions, agent_name):
        """
        Agent can see all secret if they are sharing in current room
        Agent can see all other agent's location
        Agent knows the ownship of secrets
        Agent knows whether the 
        """
        agent_loc = {}
        result = set()
        shared_value_funcs = []
        secret_loc = {}
        agent_sharing_funcs = []
        for func in functions:
            if func.name == 'agent_loc':
                result.add(func)
                agent_loc[func.parameters['?a']] = func.value
            elif func.name in ['own', 'secret_id', 'sharing_lock']:
                result.add(func)
            elif func.name == 'shared_value':
                shared_value_funcs.append(func)
            elif func.name == 'shared_loc':
                secret_loc[func.parameters['?s']] = [func.value, func]
            else:
                agent_sharing_funcs.append(func)
        for func in shared_value_funcs:
            if secret_loc[func.parameters['?s']][0] == agent_loc[agent_name]:
                result.add(func)
                result.add(secret_loc[func.parameters['?s']][1])
        # use the old jp setting, if agent didn't see the secret is sharing, agent suppose the secret is in 0
        for value in secret_loc.values():
            if value[0] != agent_loc[agent_name]:
                result.add(value[1])
        for func in agent_sharing_funcs:
            if agent_loc[agent_name] == agent_loc[func.parameters['?a']]:
                result.add(func)

        return list(result)

    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents
        
        