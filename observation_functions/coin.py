import util
import logging
from abstracts import AbstractObservationFunction
import copy

LOGGER_LEVEL = logging.DEBUG

class CoinObsFunc(AbstractObservationFunction):
    
    def get_observable_functions(self, model, functions, agent_name):
        """
        When agent is peeking, he can see everything.
        When agent is not peeking, he can see only other agent's peeking state.
        """
        is_peeking = False
        for func in functions:
            if func.name == "peeking" and func.parameters['?a'] == agent_name and func.value == 1:
                    is_peeking = True
                    break
        if is_peeking:
             return functions[:]
        else:
             return [func for func in functions if func.name == "peeking"]
    
    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents
        
        