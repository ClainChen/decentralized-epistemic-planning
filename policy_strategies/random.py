from abstracts import AbstractPolicyStrategy
from epistemic_handler.epistemic_class import Model, Action
import random
import util
import logging

LOGGER_LEVEL = logging.DEBUG

class Random(AbstractPolicyStrategy):
    """
    Agent will choose a random action
    """

    def get_policy(self, model: Model, agent_name: str) -> Action:
        successors = model.get_agent_successors(agent_name)
        return random.choice(successors) if successors else Action.stay_action(agent_name)
        