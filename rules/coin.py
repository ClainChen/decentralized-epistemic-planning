from abstracts import AbstractRules
from epistemic_handler.epistemic_class import Model, Function
import logging
import util

THIS_LOGGER_LEVEL = logging.DEBUG

class CoinRules(AbstractRules):
    def __init__(self, handler):
        self.logger = util.setup_logger(__name__, handler, logger_level=THIS_LOGGER_LEVEL)
    
    def check_functions(self, functions: list[Function]):
        """
        Always True
        """ 
        return True


                
            
