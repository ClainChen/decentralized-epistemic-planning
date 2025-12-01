from abstracts import AbstractRules
from dep.epistemic_class import Model, Function, Condition
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class GridRules(AbstractRules):
    cache = {}
    
    def check_functions(self, functions: list[Function]):
        """
        1. Survivors cannot in the same location
        2. if share_lock = 1, then there must have one and only one agent in sharing state
        3. if share_lock = 0, then there must be no agent in sharing state
        """

        survivors_loc = []
        share_lock = 0
        sharing = []
        for func in functions:
            if func.name == "survivor_loc":
                survivors_loc.append(func.value)
            elif func.name == "share_lock":
                share_lock = func.value
            elif func.name == "sharing":
                sharing.append(func.value)
        
        if share_lock == 1:
            if sum(sharing) != 1:
                return False
        else:
            if sum(sharing) != 0:
                return False
        
        return len(survivors_loc) == len(set(survivors_loc))