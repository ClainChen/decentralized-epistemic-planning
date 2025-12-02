import util
import logging
from abstracts import AbstractObservationFunction


LOGGER_LEVEL = logging.DEBUG

class MuddyChildrenObsFunc(AbstractObservationFunction):

    def get_observable_functions(self, functions, agent_name, all_funcs, ontic_functions):
        """
        Agent can see all other muddy children except itself
        Agent think itself is muddy if the question has been asked more than the number of observed muddy children.
        We assume teacher can see everything.
        """
        if agent_name == "t":
            return functions[:]
        
        result = set()
        muddies = 0
        for func in functions:
            if func.name == "muddy":
                if func.parameters['?a'] != agent_name:
                    result.add(func)
                    muddies += func.value
                else:
                    self_muddy = func
            elif func.name == "number_of_questions":
                num_questions = func.value
                result.add(func)
            else:
                result.add(func)
        if num_questions > muddies:
            result.add(self_muddy)
        
        return list(result)
    
    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents
        
        