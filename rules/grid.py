from abstracts import AbstractRules
from dep.epistemic_class import Model, Function, Condition
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG


class GridRules(AbstractRules):
    cache = {}

    def check_functions(self, functions: list[Function]):
        ff: util.QuickQueryFunctions = util.QuickQueryFunctions.build_qqf(functions)
        survivor_loc = ff.get_by_name('survivor_loc')
        sloc = [f.value for f in survivor_loc]
        movable = ff.get_by_name('movable')

        if any(f == 'r0' for f in sloc):
            return False

        for f in movable:
            agt = f.parameters['?a']
            if f.value == 1 and ff.get('agent_loc', {'?a': agt}) == 'r0':
                return False

        return len(sloc) == len(set(sloc))
