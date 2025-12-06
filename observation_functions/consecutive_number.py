import util
import logging
from abstracts import AbstractObservationFunction

LOGGER_LEVEL = logging.DEBUG

class ConsecutiveNumberObsFunc(AbstractObservationFunction):
    def get_observable_functions(self, functions, agent_name, all_funcs, ontic_functions):
        agent_num = -1
        min = -1
        max = -1
        agent_know = []
        for func in functions:
            if func.name == "agent_number" and func.parameters['?a'] == agent_name:
                agent_num = func.value
            elif func.name == "number_range_min":
                min = func.value
            elif func.name == "number_range_max":
                max = func.value
            elif func.name == "agent_know":
                agent_know.append(func.value)
        
        # check whether the agent knows the number
        # if the agent_num is equals to min or max, then the agent knows the number
        # if any agent knows the number, the all agents knows their number
        if agent_num == min or agent_num == max or any(v == 1 for v in agent_know):
            return functions[:]
        else:
            return [func for func in functions if not (func.name == "agent_number" and func.parameters['?a'] != agent_name)]
            
    
    def get_observable_agents(self, model, functions, agent_name):
        agents = [agent.name for agent in model.agents]
        return agents
        
    def post_process_jp(self, functions, agent_name, all_funcs, ontic_functions):
        return functions[:]