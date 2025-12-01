from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function, Condition
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class MuddyChildrenRules(AbstractRules):
    cache = {}
    
    def check_functions(self, functions: list[Function]):
        return True