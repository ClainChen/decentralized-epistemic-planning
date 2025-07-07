from epistemic_handler.epistemic_class import *
from epistemic_handler.file_parser import *
import logging
from itertools import combinations, permutations, product


PROBLEM_BUILDER_LOG_LEVEL = logging.DEBUG



class ProblemBuilder:
    def __init__(self, model: Model, handler, log_level=PROBLEM_BUILDER_LOG_LEVEL):
        self.model = model
        self.logger = util.setup_logger(__name__, handler, logger_level=log_level)

        

            