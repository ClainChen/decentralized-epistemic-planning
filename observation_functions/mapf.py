import util
import logging
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG

class MAPFObsFunc(AbstractObservationFunction):
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=LOGGER_LEVEL)
    
    def get_observable_functions(self, model, functions, agent_name):
        """
        Agent knows everything
        """
        try:
            return model.ontic_functions
        except KeyError as e:
            return False
        except Exception as e:
            self.logger.error(e)
            raise e
    
    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents
        
        