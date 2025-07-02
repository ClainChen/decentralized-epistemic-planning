from abc import ABC, abstractmethod
from pddl_handler.epistemic_class import Model, Agent, Function

class AbsractObservationFunction(ABC):
    def __init__(self, handler, logger_level):
        self.logger = None

    @abstractmethod
    def get_observable_functions(self, model: Model, agent: Agent) -> list[Function]:
        pass