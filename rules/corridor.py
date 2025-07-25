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

        # get the location of the agent and item
        agent_loc = {}
        item_loc = {}
        for function in functions:
            if function.name == 'agent_loc':
                if function.parameters['?a'] not in agent_loc:
                    agent_loc[function.parameters['?a']] = function.value
                else:
                    return False
            if function.name == 'item_loc':
                if function.parameters['?i'] not in item_loc:
                    item_loc[function.parameters['?i']] = function.value
                else:
                    return False

        
        for function in functions:
            # check 1, if 1 holds, then 2, 3 must hold
            if function.name == "hold_by" and function.value == 1:
                # false if agent is not at the same locatio as item
                if (function.parameters['?a'] in agent_loc
                    and function.parameters['?i'] in item_loc
                    and agent_loc[function.parameters['?a']] != item_loc[function.parameters['?i']]):
                    return False
                for function2 in functions:
                    if (function2.name == "holding" 
                        and (function2.value == 0 
                             and function2.parameters['?a'] == function.parameters['?a'])
                        ):
                        return False
                    elif (function2.name == "is_free" 
                          and (function2.value == 1 
                               and function2.parameters['?i'] == function.parameters['?i'])
                        ):
                        return False
                    elif (function2.name == "hold_by"
                          and (
                              (function2.value == 1
                               and function2.parameters['?i'] != function.parameters['?i']
                               and function2.parameters['?a'] == function.parameters['?a'])
                           or (function2.value == 1
                               and function2.parameters['?i'] == function.parameters['?i']
                               and function2.parameters['?a'] != function.parameters['?a'])
                               )
                        ):
                        return False
        
        return True

    