import logging
import inspect
import re
from file_parser import *
from epistemic_handler.epistemic_class import *
import importlib.util
from pathlib import Path

BIG_DIVIDER = "=================\n"
MEDIUM_DIVIDER = "*****************\n"
SMALL_DIVIDER = "-----------------\n"

MODEL_FOLDER_PATH = "models/"
STRATEGY_FOLDER_PATH = "policy_strategies/"


class ClassNameFormatter(logging.Formatter):
    def format(self, record):
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == record.funcName:
                arg_info = inspect.getargvalues(frame)
                if 'self' in arg_info.locals:
                    instance = arg_info.locals['self']
                    record.classname = instance.__class__.__name__
                else:
                    record.classname = '_'
                break
            frame = frame.f_back
        
        return super().format(record)


def setup_logger_handlers(log_filename, log_mode='a', c_display=False, c_logger_level=logging.INFO):
    f_handler = logging.FileHandler(log_filename, mode=log_mode)
    c_handler = logging.StreamHandler()
    c_format = ClassNameFormatter('%(levelname)s - %(name)s.%(classname)s.%(funcName)s:\n%(message)s')
    f_format = ClassNameFormatter('%(asctime)s %(levelname)s - %(name)s.%(classname)s.%(funcName)s:\n%(message)s')
    # f_format = logging.Formatter('%(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    # default handler level are info for terminal output
    # and debug for the log output
    c_handler.setLevel(c_logger_level)
    f_handler.setLevel(logging.DEBUG)

    # if the logger exist, it does not create a new one
    handlers = [f_handler]
    if c_display:
        handlers.append(c_handler)
    return handlers


def setup_logger(name, handlers=[], logger_level=logging.INFO):
    """To setup as many loggers as you want"""
    logger = logging.getLogger(name)
    logger.handlers = handlers
    logger.setLevel(logger_level)

    return logger


def regex_search(regex, string, logger=None):
    result = re.findall(regex, string, re.M)
    if logger and not result :
        logger.error(f"result not found: \"{regex}\"")
        raise Exception(f"result not found: \"{regex}\"")
    return result

def regex_match(regex, string, logger=None):
    result = re.match(regex, string, re.M)
    if logger and not result :
        logger.error(f"result not found: \"{regex}\"")
        raise Exception(f"result not found: \"{regex}\"")
    return True if result else False

def swap_param_orders(function_schema: FunctionSchema, variable: ParsingVariable):
    new_param_orders = variable.parameters
    old_param_orders = list(function_schema.require_parameters.keys())
    for old, new in zip(old_param_orders, new_param_orders):
        if old != new:
            function_schema.require_parameters[new] = function_schema.require_parameters.pop(old)

def check_duplication(list: list | tuple):
    return len(list) != len(set(list))

def load_observation_function(observation_function_path: str, logger):
    from abstracts import AbstractObservationFunction
    path = Path(observation_function_path)
    module_name = path.stem

    spec = importlib.util.spec_from_file_location(f"{module_name}_observation_function", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    valid_classes = [cls for cls in module.__dict__.values() 
                     if inspect.isclass(cls) 
                     and issubclass(cls, AbstractObservationFunction)
                     and cls != AbstractObservationFunction]

    if not valid_classes:
        logger.error(f"No valid observation function class found in {path}")
        raise ValueError(f"file {path} do not have a subclass of {AbstractObservationFunction.__name__}")
    
    return valid_classes[0]

def load_policy_strategy(policy_strategy_path: str, logger):
    from abstracts import AbstractPolicyStrategy
    try:
        path = Path(policy_strategy_path)
        module_name = path.stem

        spec = importlib.util.spec_from_file_location(f"{module_name}_strategy", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        valid_classes = [cls for cls in module.__dict__.values() 
                        if inspect.isclass(cls) 
                        and issubclass(cls, AbstractPolicyStrategy)
                        and cls != AbstractPolicyStrategy]

        if not valid_classes:
            logger.error(f"No valid strategy class found in {path}")
            raise ValueError(f"file {path} do not have a subclass of {AbstractPolicyStrategy.__name__}")
    
        return valid_classes[0]
    except:
        logger.error(f"Failed to load strategy class from {path}")
        raise ValueError(f"Failed to load strategy class from {path}")


def compare_values(a: int | str, b: int | str, strategy: ConditionOperator) -> bool:
    strategies = {
        ConditionOperator.EQUAL: lambda a, b: a == b,
        ConditionOperator.NOT_EQUAL: lambda a, b: a != b,
        ConditionOperator.GREATER: lambda a, b: a > b,
        ConditionOperator.GREATER_EQUAL: lambda a, b: a >= b,
        ConditionOperator.LESS: lambda a, b: a < b,
        ConditionOperator.LESS_EQUAL: lambda a, b: a <= b
    }

    if strategy not in strategies:
        raise ValueError(f"strategy {strategy} is not supported")
    if not isinstance(a, type(b)) or (isinstance(a, str) and strategy not in ["==", "!="]):
        raise ValueError(f"strategy {strategy} is not supported for type {type(a)} and {type(b)}")
    return strategies[strategy](a, b)

def effect_values(a: int | str, b: int | str, strategy: EffectType) -> int | str:
    strategies = {
        EffectType.ASSIGN: lambda a, b: b,
        EffectType.INCREASE: lambda a, b: a + b,
        EffectType.DECREASE: lambda a, b: a - b
    }
    
    if strategy not in strategies:
        raise ValueError(f"strategy {strategy} is not supported")
    if not isinstance(a, type(b)) or (isinstance(a, str) and strategy != EffectType.ASSIGN):
        raise ValueError(f"strategy {strategy} is not supported for string value")
    return strategies[strategy](a, b)

def check_in_range(function: Function):
    if isinstance(function.range, list):
        return function.value in range
    else:
        min, max = function.range
        return min <= function.value <= max

def is_valid_action(functions: list[Function], action: Action) -> bool:
    """
    to check whether the action is valid in realm of given functions
    """
    for condition in action.pre_condition:
        # TODO: 这里要加关于epistemic条件的判断
        checking_function = util.get_function_with_locator(functions, condition.condition_function_locator)
        if checking_function is None:
            return False
        if condition.value is not None:
            if not compare_values(checking_function.value, condition.value, condition.condition_operator):
                return False
        else:
            target_function = get_function_with_locator(functions, condition.target_function_locator)
            if target_function is None or checking_function.value != target_function.value:    
                return False
        
    return True

def get_function_with_locator(functions: list[Function], locator: FunctionLocator):
    """
    get the function with the given locator
    """
    for function in functions:
        if function.name == locator.name and list(function.parameters.values()) == list(locator.parameters.values()):
            return function
    return None