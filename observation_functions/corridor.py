import util
import logging
from epistemic_handler.epistemic_class import Model, Agent, Function
import copy
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG

class CorridorObsFunc(AbstractObservationFunction):
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=LOGGER_LEVEL)
    
    def get_observable_functions(self, model: Model, agent_name: str) -> dict[str, list[Function]]:
        """
        Get all observable functions for an agent, include the belief to other agents\n
        
        Agent's own observable functions:\n
        1. agent knows all functions in the same room with them.
        2. agent knows all other agent's room and all item's room.
        3. if agent knows an item is in the same room with agent, then he knows whether or not this item is holding by the agent.
        4. if all items are in the same room with agent, then agent knows all agents in another room is not holding any item.

        Agent's belief to other agents:\n
        1. if agent A is in the same room as agent B, then agent A belief agent B holding the same observation functions as themself.
        2. if agent A is not in the same room as agent B, then agent A only belief agent B holding observation functions about all entities' location.
        """
        agent = model.get_agent_by_name(agent_name)
        observable_functions = []
        agent_at_room = {}
        item_at_room = {}
        all_item_in_same_room = True
        try:
            for function in model.ontic_functions:
                if function.name == 'agent_loc':
                    agent_at_room[function.parameters['?a']] = function.value

            for function in model.ontic_functions:
                if function.name == 'item_loc':
                    item_at_room[function.parameters['?i']] = function.value
                    if function.value != agent_at_room[agent.name]:
                        all_item_in_same_room = False

            # self.logger.debug(f"agent at room: {agent_at_room}\nitem at room: {item_at_room}")

            for function in model.ontic_functions:
                if function.name == 'agent_loc':
                    observable_functions.append(copy.deepcopy(function))

                elif function.name == 'item_loc':
                    observable_functions.append(copy.deepcopy(function))

                elif function.name == 'holding':
                    # check whether the holding agent is at the same room as current agent
                    if (all_item_in_same_room
                        or agent_at_room[function.parameters['?a']] == agent_at_room[agent.name]):
                        observable_functions.append(copy.deepcopy(function))

                elif function.name == 'hold_by':
                    # check whether the holding agent is at the same room as current agent
                    if (all_item_in_same_room
                        or item_at_room[function.parameters['?i']] == agent_at_room[agent.name]):
                        observable_functions.append(copy.deepcopy(function))

                elif function.name == 'is_free':
                    # check whether the item is at the same room as current agent
                    if item_at_room[function.parameters['?i']] == agent_at_room[agent.name]:
                        observable_functions.append(copy.deepcopy(function))
            
            result = {}
            belief_in_another_room = []
            for function in observable_functions:
                if function.name in ['agent_loc', 'item_loc']:
                    belief_in_another_room.append(function)
                elif (all_item_in_same_room 
                      and function.name == 'holding'):
                    func_agt_name = function.parameters['?a']
                    if agent_at_room[func_agt_name] != agent_at_room[agent.name]:
                        belief_in_another_room.append(function)
                elif (all_item_in_same_room 
                      and function.name == 'hold_by'):
                    func_agt_name = function.parameters['?a']
                    if agent_at_room[func_agt_name] != agent_at_room[agent.name]:
                        belief_in_another_room.append(function)
            for other_agent in model.agents:
                if agent_at_room[other_agent.name] == agent_at_room[agent.name]:
                    result[other_agent.name] = observable_functions
                else:
                    result[other_agent.name] = belief_in_another_room
            return result
        except Exception as e:
            self.logger.error(e)
            raise e
        