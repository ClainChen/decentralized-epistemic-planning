from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class CoinRules(AbstractRules):
    
    def check_functions(self, functions: list[Function]):
        """
        Always True
        """ 
        return True


                
            
