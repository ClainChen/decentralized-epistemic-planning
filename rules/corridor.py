from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class CorridorRules(AbstractRules):
    def __init__(self, handler):
        self.logger = util.setup_logger(__name__, handler, logger_level=THIS_LOGGER_LEVEL)
    
    def check_functions(self, functions: list[Function]):
        """
        1. if hold_by ?i ?a = 1, then holding ?a = 1 and is_free ?i = 0, and this agent cannot hold any other item, and this item cannot be held by any other agent.
        2. if holding ?a = 1, then there must have one hold_by ?i ?a = 1
        3. if is_free ?i = 1, then there must all hold_by ?i ?a = 0
        4. if agent_loc ?a = ?v and hold_by ?i ?a = 1, then item_loc ?i = ?v
        5. if item_loc ?i = ?v and hold_by ?i ?a = 1, then agent_loc ?a = ?v
        """

        agent_loc_funcs = []
        item_loc_funcs = []
        holding_funcs = []
        hold_by_funcs = []
        is_free_funcs = []
        for func in functions:
            if func.name == 'agent_loc':
                agent_loc_funcs.append(func)
            elif func.name == 'item_loc':
                item_loc_funcs.append(func)
            elif func.name == 'holding':
                holding_funcs.append(func)
            elif func.name == 'hold_by':
                hold_by_funcs.append(func)
            elif func.name == 'is_free':
                is_free_funcs.append(func)

        # get the location of the agent and item
        agent_loc = {}
        agents = []
        item_loc = {}
        items = []

        for function in agent_loc_funcs:
            if function.parameters['?a'] not in agents:
                agents.append(function.parameters['?a'])
            if function.parameters['?a'] not in agent_loc:
                agent_loc[function.parameters['?a']] = function.value
            else:
                return False
                

        for function in item_loc_funcs:
            if function.parameters['?i'] not in items:
                items.append(function.parameters['?i'])
            if function.parameters['?i'] not in item_loc:
                item_loc[function.parameters['?i']] = function.value
            else:
                return False


        if len(agents) != len(agent_loc) or len(items) != len(item_loc):
            return False
        
        # 如果agent holding为true，则必然有一个hold by agent item为true
        for holding_func in holding_funcs:
            count_hold_by = 0
            for hold_by_func in hold_by_funcs:
                if (holding_func.parameters['?a'] == hold_by_func.parameters['?a']
                    and hold_by_func.value == 1):
                    count_hold_by += 1
            if (count_hold_by > 1
                or (holding_func.value == 1 and count_hold_by == 0)
                or (holding_func.value == 0 and count_hold_by != 0)):
                return False


        #如果hold by为true，则:
        # 1. agent和item必然在同一个房间中。
        # 2. agent必然holding item = 1
        # 3. item必然is free = 0
        for hold_by_func in hold_by_funcs:
            if hold_by_func.value == 1:
                if agent_loc[hold_by_func.parameters['?a']] != item_loc[hold_by_func.parameters['?i']]:
                    return False
                for holding_func in holding_funcs:
                    if (holding_func.parameters['?a'] == hold_by_func.parameters['?a']
                        and holding_func.value == 0):
                        return False
                for is_free_func in is_free_funcs:
                    if (is_free_func.parameters['?i'] == hold_by_func.parameters['?i']
                        and is_free_func.value == 1):
                        return False
        
        # 如果is free为true，则不会有任何agent持有该物品
        for is_free_func in is_free_funcs:
            count = 0
            for hold_by_func in hold_by_funcs:
                if (hold_by_func.parameters['?i'] == is_free_func.parameters['?i']
                    and hold_by_func.value == 1):
                    count += 1
            if ((is_free_func.value == 1 and count != 0)
                or (is_free_func.value == 0 and count == 0)):
                return False

        return True

    