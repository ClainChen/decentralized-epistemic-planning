from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class MAPFRules(AbstractRules):
    
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
        room_id_functions = []
        for func in functions:
            if func.name == "connected":
                connected_functions.append(func)
            elif func.name == "room_id":
                room_id_functions.append(func)
            elif func.name == "agent_at":
                agent_at_functions.append(func)
            elif func.name == "room_has_agent":
                room_has_agent_functions.append(func)
        
        # check no duplicate room id
        s = set()
        for func in room_id_functions:
            before = len(s)
            s.add(func.value)
            if len(s) == before:
                return False

        # 3. 4.
        checked_connected = set()
        for func1 in connected_functions:
            r11 = func1.parameters['?r1']
            r12 = func1.parameters['?r2']
            if r11 == r12:
                if func1.value == 1:
                    return False
                else:
                    continue
            checked_connected.add(func1)
            for func2 in connected_functions:
                if func2 in checked_connected:
                    continue
                r21 = func2.parameters['?r1']
                r22 = func2.parameters['?r2']
                if r11 == r22 and r12 == r21 and func1.value != func2.value:
                    return False 

        # 1, 2
        s1 = set()
        for func in agent_at_functions:
            before = len(s1)
            s1.add(func.value)
            if len(s1) == before:
                return False

        # 2
        s2 = set()
        for func in room_has_agent_functions:
            before = len(s2)
            if func.value == 1:
                s2.add(func.parameters['?r'])
                if len(s2) == before:
                    return False
        if s1 != s2:
            return False
            
        
        return True


                
            
