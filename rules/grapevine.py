from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class GrapevineRules(AbstractRules):
    
    def check_functions(self, functions: list[Function]):
        """
        One and only one of the agent can share the secret in a timestamp
        """ 
        some_one_sharing_secret = False
        agent_sharing = {}
        agent_loc = {}
        secret_info = {} # secret: [sharing_value, loc]
        for func in functions:
            if func.name == 'agent_loc':
                if func.value not in [1,2]:
                    return False
                agent_loc[func.parameters['?a']] = func.value
            elif func.name == 'secret_id':
                if func.parameters['?s'] != func.value:
                    return False
                secret_info[func.value] = {}
            elif func.name == 'shared_value':
                if func.value not in ['t', 'f']:
                    return False
                secret_info[func.parameters['?s']]['value'] = func.value
            # if shared_loc != 0, it means someone is sharing the secret in the room
            elif func.name == 'shared_loc':
                secret_info[func.parameters['?s']]['loc'] = func.value
            elif func.name == 'own':
                pass
            elif func.name == 'sharing_lock':
                some_one_sharing_secret = func.value == 1
            elif func.name == 'agent_sharing':
                agent_sharing[func.parameters['?a']] = func.value
        
        # for f in functions:
        #     print(f)
        # if sharing_loc == 1, mean one and only one agent must sharing the secret. If not, means nobody is sharing the secret.
        if some_one_sharing_secret:
            loc_sum = [value['loc'] for value in secret_info.values() if value['loc'] != 0]
            if len(loc_sum) == 0 or len(loc_sum) > 1 or sum(loc_sum) > 2:
                # print(f"some one is sharing the secret but secret_loc has problem: {len(loc_sum)} {sum(loc_sum)}")
                return False
            sharing_sum = sum([1 if v != 'none' else 0 for v in agent_sharing.values()])
            if sharing_sum == 0 or sharing_sum > 1:
                # print(f"some one is sharing the secret but agent_sharing has problem: {sharing_sum}")
                return False

        # if secret_loc == 0 but the shared_value == 'f', it is wrong
        for values in secret_info.values():
            if values['value'] != 't' and values['loc'] == 0:
                # print(f"secret_loc == 0 but the shared_value == 'f'")
                return False
        
        # if the agent sharing a value but the value's secret_loc is not in the same room, it is wrong
        for agt, agt_loc in agent_loc.items():
            s = agent_sharing[agt]
            if s != 'none':
                if secret_info[s]['loc'] != agt_loc:
                    # print("the agent sharing a value but the value's secret_loc is not in the same room")
                    for f in functions:
                        print(f"{f}")
                    return False

        return True


                
            
