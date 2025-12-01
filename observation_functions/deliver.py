import util
import logging
from epistemic_handler.epistemic_class import Model, Agent, Function
import cache
from abstracts import AbstractObservationFunction
from collections import defaultdict

LOGGER_LEVEL = logging.DEBUG

class DeliverObsFunc(AbstractObservationFunction):
    # @cache.obs_func_cache_decorator
    def get_observable_functions(self, model: Model, functions: list[Function], agent_name: str) -> list[Function]:
        """
        1. agent能看见所有与自己相关的functions
        2. agent能看见所有与自己当前所在房间相关的functions
        3. agent能看见所有与自己在同一房间中其他agent有关的functions
        4. agent能看见所有connected和room_id functions
        5. 当agent能看见所有item，则知道所有agent的holding和hold_by functions
        """
        observable_functions = set()
        agent_loc_funcs = []
        item_loc_funcs = []
        connected_funcs = set()
        room_id_funcs = set()
        holding_funcs = []
        hold_by_funcs = []
        is_free_funcs = []
        for func in functions:
            if func.name == 'agent_loc':
                agent_loc_funcs.append(func)
            elif func.name == 'item_loc':
                item_loc_funcs.append(func)
            elif func.name == 'connected':
                connected_funcs.add(func)
            elif func.name == 'room_id':
                room_id_funcs.add(func)
            elif func.name == 'holding':
                holding_funcs.append(func)
            elif func.name == 'hold_by':
                hold_by_funcs.append(func)
            elif func.name == 'is_free':
                is_free_funcs.append(func)
        
        agent_at = defaultdict(str)
        for func in agent_loc_funcs:
            agent_at[func.parameters['?a']] = func.value

        item_at = defaultdict(str)
        for func in item_loc_funcs:
            item_at[func.parameters['?i']] = func.value

        try:
            # agent知道所有与自己有关的functions
            for func in functions:
                if func.name == 'agent_loc':
                    if func.value == agent_at[agent_name]:
                        observable_functions.add(func)
                elif func.name == 'item_loc':
                    if func.value == agent_at[agent_name]:
                        observable_functions.add(func)
                elif func.name in ['connected', 'room_id']:
                    observable_functions.add(func)
                elif func.name == 'hold_by':
                    if func.parameters['?a'] == agent_name:
                        observable_functions.add(func)
                    if agent_at[func.parameters['?a']] == agent_at[agent_name]:
                        observable_functions.add(func)
                    if item_at[func.parameters['?i']] == agent_at[agent_name]:
                        observable_functions.add(func)

            return list(observable_functions)
        except KeyError as e:
            util.LOGGER.error(e)
            raise e
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