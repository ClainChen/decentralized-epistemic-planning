from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action
import random
import util
import logging

LOGGER_LEVEL = logging.DEBUG

class GreedyBFS(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        if len(successors) > 1:
            self.bfs(model, agent_name)
        elif len(successors) == 1:
            return successors[0]
        else:
            return None
        
    def bfs(model: Model, agent_name: str):
        # TODO: finish bfs here
        agent_index = model.get_agent_index_by_name(agent_name)
