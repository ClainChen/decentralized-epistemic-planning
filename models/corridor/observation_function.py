import util
import logging
from pddl_handler.epistemic_class import Model, Agent, Function
import copy
from absract_observation_function import AbsractObservationFunction

OBSERVATION_LOGGER_LEVEL = logging.DEBUG

class ObservationFunction(AbsractObservationFunction):
    def __init__(self, handler):
        self.logger = util.setup_logger(__name__, handler, logger_level=OBSERVATION_LOGGER_LEVEL)
    
    def get_observable_functions(self, model: Model, agent: Agent) -> list[Function]:
        """
        1. agent knows all functions in the same room with them
        2. agent knows all other agent's room
        """
        observable_functions = []
        at_room = {}

        try:
            for function in model.ontic_functions:
                if function.name == 'agent_loc':
                    at_room[function.parameters['?a']] = function.value
                elif function.name == 'item_loc':
                    at_room[function.parameters['?i']] = function.value
            self.logger.debug(f"at room: {at_room}")

            for function in model.ontic_functions:
                if function.name == 'agent_loc':
                    observable_functions.append(copy.deepcopy(function))

                elif function.name == 'item_loc':
                    observable_functions.append(copy.deepcopy(function))

                elif function.name == 'holding':
                    # check whether the holding agent is at the same room as current agent
                    if at_room[function.parameters['?a']] == at_room[agent.name]:
                        observable_functions.append(copy.deepcopy(function))

                elif function.name == 'hold_by':
                    # check whether the holding agent is at the same room as current agent
                    if at_room[function.parameters['?i']] == at_room[agent.name]:
                        observable_functions.append(copy.deepcopy(function))

                elif function.name == 'is_free':
                    # check whether the item is at the same room as current agent
                    if at_room[function.parameters['?i']] == at_room[agent.name]:
                        observable_functions.append(copy.deepcopy(function))
            return observable_functions
        except Exception as e:
            self.logger.error(e)
            raise e
        