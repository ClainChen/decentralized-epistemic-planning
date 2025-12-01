from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function, Condition
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class ConsecutiveNumberRules(AbstractRules):
    cache = {}
    
    def check_functions(self, functions: list[Function]):
        agent_nums = []
        min = -1
        max = -1
        for func in functions:
            if func.name == "agent_number":
                agent_nums.append(func.value)
            elif func.name == "number_range_min":
                min = func.value
            elif func.name == "number_range_max":
                max = func.value
        if min >= max:
            return False
        if abs(agent_nums[0] - agent_nums[1]) != 1:
            return False
        return True