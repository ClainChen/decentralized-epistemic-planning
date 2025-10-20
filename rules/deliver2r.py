from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Function, Condition, Model
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class Deliver2rRules(AbstractRules):
    compain_cache = {}
    verification_cache = {}

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
                # util.LOGGER.info(f"Agent {function.parameters['?a']} has multiple locations")
                return False
                

        for function in item_loc_funcs:
            if function.parameters['?i'] not in items:
                items.append(function.parameters['?i'])
            if function.parameters['?i'] not in item_loc:
                item_loc[function.parameters['?i']] = function.value
            else:
                # util.LOGGER.info(f"Item {function.parameters['?i']} has multiple locations")
                return False


        if len(agents) != len(agent_loc) or len(items) != len(item_loc):
            # util.LOGGER.info("Not all agents and items have locations")
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
                # util.LOGGER.info(f"holding functions has invalid settings")
                return False


        #如果hold by为true，则:
        # 1. agent和item必然在同一个房间中。
        # 2. agent必然holding item = 1
        # 3. item必然is free = 0
        # 4. 必然不会有另一个agent正在hold同一个item
        for hold_by_func in hold_by_funcs:
            if hold_by_func.value == 1:
                if agent_loc[hold_by_func.parameters['?a']] != item_loc[hold_by_func.parameters['?i']]:
                    # util.LOGGER.info(f"hold by functions has invalid settings")
                    return False
                for hold_by_func2 in hold_by_funcs:
                    if (hold_by_func2.value == 1
                        and hold_by_func2.parameters['?a'] != hold_by_func.parameters['?a']
                        and hold_by_func2.parameters['?i'] == hold_by_func.parameters['?i']):
                        # util.LOGGER.info(f"hold by functions has invalid settings")
                        return False
                for holding_func in holding_funcs:
                    if (holding_func.parameters['?a'] == hold_by_func.parameters['?a']
                        and holding_func.value == 0):
                        # util.LOGGER.info(f"hold by functions has invalid settings")
                        return False
                for is_free_func in is_free_funcs:
                    if (is_free_func.parameters['?i'] == hold_by_func.parameters['?i']
                        and is_free_func.value == 1):
                        # util.LOGGER.info(f"hold by functions has invalid settings")
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
                # util.LOGGER.info(f"is free functions has invalid settings")
                return False

        return True
    
    def belongs_to(self, function: Function):
        # now only consider the belonging of 'hold_by'
        return function.parameters['?a']
    
    def companion_funcs(self, func: Function, model:Model) -> list[Function]:
        result = []
        if func.name == 'hold_by':
            if func.value == 1:
                cur_agt = func.parameters['?a']
                for agt in model.get_all_agent_names():
                    if agt != cur_agt:
                        func_param = {'?i': func.parameters['?i'], '?a': agt}
                        result.append(model.ALL_FUNCS.get_function(func.name, func_param, "0"))
        return result + [func]
    
    def check_valid_pair(self, cond1: Condition, cond2: Condition, model: Model):
        # related functions of condition 1
        rela1 = set(cond1.belief_sequence)
        func1 = model.ALL_FUNCS.get_function(cond1.condition_function_name, cond1.condition_function_parameters, cond1.value)
        if func1.id not in self.compain_cache:
            self.compain_cache[func1.id] = [f.id for f in self.companion_funcs(func1, model)]
        for id in self.compain_cache[func1.id]:
            func = model.ALL_FUNCS.get_function_with_id(id)
            rela1.add(self.belongs_to(func))
        
        # related functions of condition 2
        rela2 = set(cond2.belief_sequence)
        func2 = model.ALL_FUNCS.get_function(cond2.condition_function_name, cond2.condition_function_parameters, cond2.value)
        if func2.id not in self.compain_cache:
            self.compain_cache[func2.id] = [f.id for f in self.companion_funcs(func2, model)]
        for id in self.compain_cache[func2.id]:
            func = model.ALL_FUNCS.get_function_with_id(id)
            rela2.add(self.belongs_to(func))
        
        # if the included agent are different, then they are not related
        if rela1 != rela2:
            self.verification_cache[frozenset([cond1, cond2])] = True
            return True
        
        # if the included agent are the same, do further verification
        funcs = self.compain_cache[func1.id] + self.compain_cache[func2.id]
        for i in range(len(funcs) - 1):
            for j in range(i + 1, len(funcs)):
                id1 = funcs[i]
                id2 = funcs[j]
                
                fs = frozenset([id1, id2])
                if fs in self.verification_cache:
                    if self.verification_cache[fs]:
                        continue
                    return False

                func1 = model.ALL_FUNCS.get_function_with_id(id1)
                func2 = model.ALL_FUNCS.get_function_with_id(id2)
                # different agent carrying the same item
                if (func1.value == 1 and 
                    func2.value == 1 and 
                    func1.parameters['?a'] != func2.parameters['?a'] and 
                    func1.parameters['?i'] == func2.parameters['?i']):
                    self.verification_cache[fs] = False
                    return False
                
                # same agent, hi = 1 and hi = 0 in the same time
                if (func1.value != func2.value and 
                    func1.parameters['?a'] == func2.parameters['?a'] and 
                    func1.parameters['?i'] == func2.parameters['?i']):
                    self.verification_cache[fs] = False
                    return False

                self.verification_cache[fs] = True
        return True


        

