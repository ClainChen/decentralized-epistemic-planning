from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class GrapevineRules(AbstractRules):
    def __init__(self, handler):
        self.logger = util.setup_logger(__name__, handler, logger_level=THIS_LOGGER_LEVEL)
    
    def check_functions(self, functions: list[Function]):
        """
        One and only one of the agent can share the secret in a timestamp
        """ 
        some_one_sharing_secret = False
        shared_loc = []
        agent_sharing = []
        for func in functions:
            if func.name == 'agent_loc':
                if func.value not in [1,2]:
                    return False
            elif func.name == 'secret_id':
                if func.parameters['?s'] != func.value:
                    return False
            elif func.name == 'shared_value':
                if func.value not in ['t', 'f']:
                    return False
            # if shared_loc != 0, it means someone is sharing the secret in the room
            elif func.name == 'shared_loc':
                shared_loc.append(func)
            elif func.name == 'own':
                pass
            elif func.name == 'sharing_lock':
                some_one_sharing_secret = func.value == 1
            elif func.name == 'agent_sharing':
                agent_sharing.append(func)
        
        # if sharing_loc == 1, mean one and only one agent must sharing the secret. If not, means nobody is sharing the secret.
        if some_one_sharing_secret:
            if sum([func.value for func in shared_loc]) > 1:
                return False
            if sum([1 if func.value != 'none' else 0 for func in agent_sharing]) > 1:
                return False
        return True


                
            
