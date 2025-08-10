import util
import logging
from epistemic_handler.epistemic_class import Model, Agent, Function
import copy
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG

class MAPFObsFunc(AbstractObservationFunction):
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=LOGGER_LEVEL)
    
    def get_observable_functions(self, model: Model, functions: list[Function], agent_name: str) -> list[Function]:
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
        
        