import util
import logging
from abstracts import AbstractObservationFunction
import copy

LOGGER_LEVEL = logging.DEBUG

class MAPFObsFunc(AbstractObservationFunction):
    
    def get_observable_functions(self, model, functions, agent_name):
        """
        Agent knows everything
        """
        return functions[:]
    
    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents
        
        