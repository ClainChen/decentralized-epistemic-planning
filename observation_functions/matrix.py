import util
import logging
from epistemic_handler.epistemic_class import Model, Agent, Function
import copy
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG

class MatrixObsFunc(AbstractObservationFunction):
    def __init__(self, handler, logger_level=LOGGER_LEVEL):
        self.logger = util.setup_logger(__name__, handler, logger_level=LOGGER_LEVEL)
    
    def get_observable_functions(self, model: Model, functions: list[Function], agent_name: str) -> list[Function]:
        """
        1. agent_name知道一切与自己在同房间中的信息
            1.1 如果ontic functions中没有agent_name所在的房间，该agent_name只能从ontic functions获取connections信息和与自己有关的信息。
        2. 检测该ontic functions是否是对现实世界进行观测，如果是，则会判断是否所有item都与自己在同一个房间，如果所有item都与自己在同一个房间，则代表该agent_name知道任何不在该房间的其他agent都不会拿着任何item
        """
        observable_functions = set()
        agent_loc_funcs = []
        item_loc_funcs = []
        connected_funcs = []
        holding_funcs = []
        hold_by_funcs = []
        is_free_funcs = []
        for func in functions:
            if func.name == 'agent_loc':
                agent_loc_funcs.append(func)
            elif func.name == 'item_loc':
                item_loc_funcs.append(func)
            elif func.name == 'connected':
                connected_funcs.append(func)
            elif func.name == 'holding':
                holding_funcs.append(func)
            elif func.name == 'hold_by':
                hold_by_funcs.append(func)
            elif func.name == 'is_free':
                is_free_funcs.append(func)
        
        for func in connected_funcs:
            observable_functions.add(func)
        
        agent_at = {}
        for func in agent_loc_funcs:
            if func.value == 1 and func.parameters['?a'] not in agent_at:
                agent_at[func.parameters['?a']] = func.parameters['?loc']

        item_at = {}
        for func in item_loc_funcs:
            if func.value == 1 and func.parameters['?i'] not in item_at:
                item_at[func.parameters['?i']] = func.parameters['?loc']

        try:
            # agent知道所有与自己有关的functions
            for func in agent_loc_funcs + hold_by_funcs + holding_funcs:
                if func.parameters['?a'] == agent_name:
                    observable_functions.add(func)

            # agent知道所有与自己当前所在房间相关的functions
            # agent知道所有与自己在同一房间中其他agent有关的functions
            # 如果在functions中没有agent_name的位置，那么只会返回与agent_name有关的functions
            if agent_name in agent_at:
                for func in agent_loc_funcs:
                    if (func.parameters['?loc'] == agent_at[agent_name] or 
                        (func.parameters['?a'] in agent_at and
                         agent_at[func.parameters['?a']] == agent_at[agent_name])
                        ):
                        observable_functions.add(func)
                for func in item_loc_funcs:
                    if (func.parameters['?loc'] == agent_at[agent_name] or 
                        (func.parameters['?i'] in item_at and
                         item_at[func.parameters['?i']] == agent_at[agent_name])
                        ):
                        observable_functions.add(func)
                for func in hold_by_funcs:
                    if func.parameters['?a'] in agent_at and agent_at[func.parameters['?a']] == agent_at[agent_name]:
                        observable_functions.add(func)
                    elif func.parameters['?i'] in item_at and item_at[func.parameters['?i']] == agent_at[agent_name]:
                        observable_functions.add(func)
                for func in holding_funcs:
                    if func.parameters['?a'] in agent_at and agent_at[func.parameters['?a']] == agent_at[agent_name]:
                        observable_functions.add(func)
                for func in is_free_funcs:
                    if func.parameters['?i'] in item_at and item_at[func.parameters['?i']] == agent_at[agent_name]:
                        observable_functions.add(func)

            return list(observable_functions)
        except KeyError as e:
            self.logger.error(e)
            raise e
        except Exception as e:
            self.logger.error(e)
            raise e
        
        