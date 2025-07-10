import util
import logging
from epistemic_handler.epistemic_class import Model, Agent, Function
import copy
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG

class CorridorObsFunc(AbstractObservationFunction):
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=LOGGER_LEVEL)
    
    def get_observable_functions(self, ontic_functions: list[Function], agent_name: str) -> list[Function]:
        """
        Get all observable functions for an agent based on the given ontic_functions\n
        
        Agent's own observable functions:\n
        1. agent knows all functions in the same room with them.
        2. agent knows all other agent's room and all item's room.
        3. if agent knows an item is in the same room with agent, then he knows whether or not this item is holding by the agent.
        4. if all items are in the same room with agent, then agent knows all agents in another room is not holding any item.
        """
        observable_functions = []
        agent_at_room = {}
        item_at_room = {}
        all_item_in_same_room = True
        try:
            for function in ontic_functions:
                if function.name == 'agent_loc':
                    agent_at_room[function.parameters['?a']] = function.value
                    observable_functions.append(copy.deepcopy(function))

            for function in ontic_functions:
                if function.name == 'item_loc':
                    item_at_room[function.parameters['?i']] = function.value
                    observable_functions.append(copy.deepcopy(function))
                    if function.value != agent_at_room[agent_name]:
                        all_item_in_same_room = False

            # self.logger.debug(f"agent at room: {agent_at_room}\nitem at room: {item_at_room}")

            for function in ontic_functions:
                if function.name == 'holding':
                    # check whether the holding agent is at the same room as current agent
                    if (all_item_in_same_room
                        or agent_at_room[function.parameters['?a']] == agent_at_room[agent_name]):
                        observable_functions.append(copy.deepcopy(function))

                elif function.name == 'hold_by':
                    # check whether the holding agent is at the same room as current agent
                    if (agent_at_room[function.parameters['?a']] == agent_at_room[agent_name]
                        or item_at_room[function.parameters['?i']] == agent_at_room[agent_name]):
                        observable_functions.append(copy.deepcopy(function))

                elif function.name == 'is_free':
                    # check whether the item is at the same room as current agent
                    if item_at_room[function.parameters['?i']] == agent_at_room[agent_name]:
                        observable_functions.append(copy.deepcopy(function))
            
            return observable_functions
        except Exception as e:
            self.logger.error(e)
            raise e
        