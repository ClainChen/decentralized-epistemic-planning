from abc import ABC, abstractmethod
from epistemic_handler.epistemic_class import Model, Agent, Function, Action

class AbstractObservationFunction(ABC):
    """
    Abstract class for observation function\n
    1. Make sure put the new observation function file in the observation_functions folder\n
    2. Make sure to extend this class and implement the abstract method when you are defining a new observation function class, that is really important\n
    """
    def __init__(self, handler, logger_level):
        self.logger = None

    @abstractmethod
    def get_observable_functions(self, model: Model, agent_name: str) -> dict[str, list[Function]]:
        pass

class AbstractPolicyStrategy(ABC):
    """
    Abstract class for agent policy decision strategy\n
    1. Agent will only implement the strategy if the agent has the belief of all other agents' current functions. Otherwise, they will only use a greedy strategy.\n
    2. Make sure put the new policy decision strategy class in the policy_strategies folder.\n
    3. Make sure to extend this class and implement the abstract method when you are defining a new observation function class, that is really important\n
    """

    def __init__(self, handler, logger_level):
        self.logger = None

    @abstractmethod
    def get_policy(self, model: Model, agent_name: str) -> Action:
        pass


class AbstractRules(ABC):
    """
    Abstract class for rules, if you needs to use virtual world generator or problem generator, then make sure you build a rule class for specific problem.\n
    This class is used to check whether the functions of the model or a set of functions is following the rules.\n
    1. Make sure put the new rules class in the rules folder.\n
    2. Make sure to extend this class and implement the abstract method when you are defining a new rules class, that is really important\n
    """

    def __init__(self, handler, logger_level):
        self.logger = None

    def check_model(self, model: Model):
        return self.check_functions(model.ontic_functions)
    
    @abstractmethod
    def check_functions(self, functions: list[Function]) -> bool:
        pass