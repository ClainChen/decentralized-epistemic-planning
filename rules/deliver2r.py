from abstracts import AbstractRules
from dep.epistemic_class import Function, Condition, Model
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class Deliver2rRules(AbstractRules):
    compain_cache = {}
    verification_cache = {}

    def check_functions(self, functions: list[Function]):
        agent_loc_funcs = []
        item_loc_funcs = []
        hold_funcs = []
        is_free_funcs = {}
        for func in functions:
            if func.name == 'agent_loc':
                agent_loc_funcs.append(func)
            elif func.name == 'item_loc':
                item_loc_funcs.append(func)
            elif func.name == 'hold':
                hold_funcs.append(func)
            elif func.name == 'is_free':
                is_free_funcs[func.parameters['?i']] = func.value

        # get the location of the agent and item
        agent_locs = {}
        agents = []
        item_locs = {}
        items = []

        for function in agent_loc_funcs:
            if function.parameters['?a'] not in agents:
                agents.append(function.parameters['?a'])
            if function.parameters['?a'] not in agent_locs:
                agent_locs[function.parameters['?a']] = function.value
            else:
                # util.LOGGER.info(f"Agent {function.parameters['?a']} has multiple locations")
                return False
                

        for function in item_loc_funcs:
            if function.parameters['?i'] not in items:
                items.append(function.parameters['?i'])
            if function.parameters['?i'] not in item_locs:
                item_locs[function.parameters['?i']] = function.value
            else:
                # util.LOGGER.info(f"Item {function.parameters['?i']} has multiple locations")
                return False


        if len(agents) != len(agent_locs) or len(items) != len(item_locs):
            # util.LOGGER.info("Not all agents and items have locations")
            return False

        
        # if hold is true, then:
        # 1. agent and item must be in the same room.
        # 2. item must is free = 0 unless it is a nothing
        # 3. there must not have another agent holding the same item unless it is a nothing
        for hold_func in hold_funcs:
            agt = hold_func.parameters['?a']
            agt_loc = agent_locs[agt]
            item = hold_func.value
            item_loc = item_locs[item]
            item_is_free = is_free_funcs[item] == 1
            if item != 'nothing' and (item_is_free or (agt_loc != item_loc)):
                return False

        return True
