from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class MAPFRules(AbstractRules):
    def __init__(self, handler):
        self.logger = util.setup_logger(__name__, handler, logger_level=THIS_LOGGER_LEVEL)
    
    def check_functions(self, functions: list[Function]):
        """
        1. When an agent in a room, then no other agent can be in the same room
        2. When a room has agent, then only one agent can be in that room
        3. When a connect a b = 1 holds, then connect b a = 1 holds, same in = 0
        4. connect a a = 0 always holds, = 1 always not holds
        """
        connected_functions = []
        agent_at_functions = []
        room_has_agent_functions = []
        for func in functions:
            if func.name == "connected":
                connected_functions.append(func)
            elif func.name == "agent_at":
                if func.value == 1:
                    agent_at_functions.append(func)
            elif func.name == "room_has_agent":
                room_has_agent_functions.append(func)
        
        # 3. 4.
        checked_connected = set()
        for func in connected_functions:
            checked_connected.add(func)
            if func in checked_connected:
                continue

            r1 = func.parameters['?r1']
            r2 = func.parameters['?r2']
            v1 = func.value
            if r1 == r2:
                if v1 == 0:
                    continue
                else:
                    return False
            
            for func2 in connected_functions:
                if func2 in checked_connected:
                    continue
                if (func2.parameters['?r1'] == r2
                    and func2.parameters['?r2'] == r1):
                    if func2.value == v1:
                        checked_connected.add(func2)
                        break
                    else:
                        return False

        # 1
        for func in agent_at_functions:
            agt = func.parameters['?a']
            room = func.parameters['?r']
            for func2 in agent_at_functions:
                if (func2.parameters['?r'] != room
                    and func2.parameters['?a'] == agt):
                    return False
                elif (func2.parameters['?r'] == room
                      and func2.parameters['?a'] != agt):
                     return False

        # 2
        for func in room_has_agent_functions:
            room = func.parameters['?r']
            if func.value == 0:
                for func2 in agent_at_functions:
                    if (func2.parameters['?r'] == room
                        and func2.value == 1):
                        return False
            else:
                count = 0
                for func in agent_at_functions:
                    if func.parameters['?r'] == room:
                        count += 1
                        if count > 1:
                            return False
        
        return True


                
            
