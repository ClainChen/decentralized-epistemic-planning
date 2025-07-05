from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Agent, Function, Action
import random
import util
import logging

LOGGER_LEVEL = logging.DEBUG

class Random(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=logger_level)

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        return random.choice(successors) if successors else None