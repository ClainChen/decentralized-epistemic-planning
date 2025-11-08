from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class DeliverRules(AbstractRules):
    
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
        room_id_funcs = []
        holding_funcs = []
        hold_by_funcs = []
        is_free_funcs = []
        for func in functions:
            if func.name == 'agent_loc':
                agent_loc_funcs.append(func)
            elif func.name == 'item_loc':
                item_loc_funcs.append(func)
            elif func.name == 'room_id':
                room_id_funcs.append(func)
            elif func.name == 'holding':
                holding_funcs.append(func)
            elif func.name == 'hold_by':
                hold_by_funcs.append(func)
            elif func.name == 'is_free':
                is_free_funcs.append(func)

        # check duplicate room id
        s = set()
        for func in room_id_funcs:
            before = len(s)
            s.add(func.value)
            if len(s) == before:
                return False
        
        agent_loc = {func.parameters['?a']: func.value for func in agent_loc_funcs}
        item_loc = {func.parameters['?i']: func.value for func in item_loc_funcs}


        # If agent holding is true, there must have one hold by agent item is true
        for holding_func in holding_funcs:
            count_hold_by = 0
            for hold_by_func in hold_by_funcs:
                if (holding_func.parameters['?a'] == hold_by_func.parameters['?a']
                    and hold_by_func.value == 1):
                    count_hold_by += 1
            if (count_hold_by > 1
                or (holding_func.value == 1 and count_hold_by == 0)
                or (holding_func.value == 0 and count_hold_by == 1)):
                return False

       
        # if hold by is true, then:
        # 1. agent and item must be in the same room.
        # 2. agent must be holding item = 1
        # 3. item must is free = 0
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
        
        # if is free is true, then no agent holds the item
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




