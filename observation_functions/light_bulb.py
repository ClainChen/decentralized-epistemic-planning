from util import QuickQueryFunctions
import logging
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG


class LightBulbObsFunc(AbstractObservationFunction):

    def get_observable_functions(self, functions, agent_name, all_funcs, ontic_functions):
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

        ff: QuickQueryFunctions = QuickQueryFunctions.build_qqf(functions)
        # ontic_ff: QuickQueryFunctions = QuickQueryFunctions.build_qqf(ontic_functions)
        result = set(f for f in functions if f.name in ['agent_id',
                                                        'light_id',
                                                        'bs_id',
                                                        'tell_lock',
                                                        'telling',
                                                        'a_observable',
                                                        'observable',
                                                        'change_lock',
                                                        'connected'])

        if agent_name == 'a':
            for func in functions:
                if func.name == 'button_state':
                    result.add(func)
                elif func.name == 'light_state':
                    """
                    1. if a is observable to this light, then a knows that
                    """
                    light = func.parameters['?l']
                    if ff.get('a_observable', {'?l': light}) == 1:
                        result.add(func)
        elif agent_name in ['b', 'c']:
            for func in functions:
                if func.name == 'light_state':
                    """
                    1. if b or c is observable to this light, then a knows that
                    """
                    light = func.parameters['?l']
                    if ff.get('observable', {'?l': light}) == agent_name:
                        result.add(func)

        return list(result)

    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents

    def post_process_jp(self, functions, agent_name, all_funcs, ontic_functions):
        return functions[:]
