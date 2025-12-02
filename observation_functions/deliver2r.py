import util
import logging
from dep.epistemic_class import Model, Function
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG

class Deliver2rObsFunc(AbstractObservationFunction):
    def get_observable_functions(self, functions: list[Function], agent_name: str, all_funcs, ontic_functions) -> list[Function]:
        observable_functions = set()
        agent_at_room = {}
        item_at_room = {}
        all_item_in_same_room = True
        try:
            for function in functions:
                if function.name == 'agent_loc':
                    agent_at_room[function.parameters['?a']] = function.value
                    observable_functions.add(function)
                elif function.name == 'item_id':
                    observable_functions.add(function)

            for function in functions:
                if function.name == 'item_loc':
                    item_at_room[function.parameters['?i']] = function.value
                    observable_functions.add(function)
                    if function.parameters['?i'] != 'nothing' and function.value != agent_at_room[agent_name]:
                        all_item_in_same_room = False

            # util.LOGGER.debug(f"agent at room: {agent_at_room}\nitem at room: {item_at_room}")

            for function in functions:
                if function.name == 'hold':
                    # check whether the holding agent is at the same room as current agent
                    if (all_item_in_same_room
                        or agent_at_room[function.parameters['?a']] == agent_at_room[agent_name]):
                        observable_functions.add(function)

                elif function.name == 'is_free':
                    # check whether the item is at the same room as current agent
                    if (function.parameters['?i'] == 'nothing' or
                            item_at_room[function.parameters['?i']] == agent_at_room[agent_name]):
                        observable_functions.add(function)
            
            return list(observable_functions)
        except KeyError as e:
            return False
        except Exception as e:
            util.LOGGER.error(e)
            raise e

    def get_observable_agents(self, model, functions, agent_name):
        agent_room = {}
        for func in functions:
            if func.name == 'agent_loc':
                agent_room[func.parameters['?a']] = func.value
        current_agent_room = agent_room[agent_name]
        return [agent for agent, room in agent_room.items() if room == current_agent_room]
        
        