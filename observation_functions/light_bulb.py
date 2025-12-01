import util
import logging
from abstracts import AbstractObservationFunction
import cache

LOGGER_LEVEL = logging.DEBUG

class LightBulbObsFunc(AbstractObservationFunction):
    # @cache.obs_func_cache_decorator
    def get_observable_functions(self, model, functions, agent_name):
        """
        - Common Knowledge: agent_id, light_id, tell_lock, telling, a_observable
        1. Agent A:
            - knows the state of all buttons
            - knows the light state if a_observable ?l is true
        2. Agent B:
            - knows the state of light 1 and 2
        3. Agent C:
            - knows the state of light 3 and 4
        """
        if agent_name == 'external':
            return functions[:]
        result = [f for f in functions if f.name in ['agent_id', 'light_id', 'bs_id', 'tell_lock', 'telling', 'a_observable', 'observable']]

        light_states = {}
        button_states = {}
        button_light_relation = {}
        connect_relation = {}
        for func in model.ontic_functions:
            if func.name == 'light_state':
                light_states[func.parameters['?l']] = func
            elif func.name == 'button_light_state':
                button = func.parameters['?b']
                if button not in button_light_relation:
                    button_light_relation[button] = {}
                button_light_relation[button][func.parameters['?bs']] = func.value
            elif func.name == 'connected':
                connect_relation[func.parameters['?b']] = func.value
            elif func.name == 'button_state':
                button_states[func.parameters['?b']] = func

        """
        The state of the light is determined by the button_light_relation:
        (light_state ?l) = (button_light_state ?b ?bs) where (connected ?b) = ?l and (button_state ?b) = ?bs
        """
        if agent_name == 'a':
            a_observable = {}
            for func in functions:
                if func.name == 'button_state':
                    result.append(func)
                elif func.name == 'a_observable':
                    a_observable[func.parameters['?l']] = func.value
            
            for l, is_observable in a_observable.items():
                if is_observable == 1:
                    result.append(light_states[l])
            return result
        else:
            for func in functions:
                if func.name == 'observable':
                    if func.value == agent_name:
                        result.append(light_states[func.parameters['?l']])
            return result
    
    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents
        
        