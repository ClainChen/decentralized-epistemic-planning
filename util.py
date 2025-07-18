import logging
import inspect
import re
from epistemic_handler.file_parser import *
from epistemic_handler.epistemic_class import *
import importlib.util
from pathlib import Path
import random

BIG_DIVIDER = "=================\n"
MEDIUM_DIVIDER = "*****************\n"
SMALL_DIVIDER = "-----------------\n"

MODEL_FOLDER_PATH = "models/"
OBS_FUNC_FOLER_PATH = "observation_functions/"
STRATEGY_FOLDER_PATH = "policy_strategies/"
RULES_FOLDER_PATH = "rules/"


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

def load_rules(rules_path: str, logger):
    from abstracts import AbstractRules
    try:
        path = Path(rules_path)
        module_name = path.stem

        spec = importlib.util.spec_from_file_location(f"{module_name}_strategy", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        valid_classes = [cls for cls in module.__dict__.values() 
                        if inspect.isclass(cls) 
                        and issubclass(cls, AbstractRules)
                        and cls != AbstractRules]

        if not valid_classes:
            logger.error(f"No valid rules class found in {path}")
            raise ValueError(f"file {path} do not have a subclass of {AbstractRules.__name__}")
    
        return valid_classes[0]
    except:
        logger.error(f"Failed to load rules class from {path}")
        raise ValueError(f"Failed to load rules class from {path}")

def compare_condition_values(a: int | str, b: int | str, strategy: ConditionOperator) -> bool:
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

def update_effect_value(a: int | str, b: int | str, strategy: EffectType) -> int | str:
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
        return function.value in function.range
    else:
        min, max = function.range
        return min <= function.value <= max

def is_valid_action(functions: list[Function], action: Action, agent_history: list[list[Function]] = [], obs_func = None, is_ontic_checking: bool = True) -> bool:
    """
    to check whether the action is valid in realm of given functions
    """
    if action is None:
        return True
    for condition in action.pre_condition:
        if is_ontic_checking:
            if len(condition.belief_sequence) == 0:
                if not check_regular_condition(condition, functions):
                    return False
        else:
            if not check_condition(condition, functions, agent_history, obs_func):
                return False
    return True

def check_condition(condition: Condition, functions: list[Function], agent_history: list[list[Function]], obs_func):
    if len(condition.belief_sequence) == 0:
        return check_regular_condition(condition, functions)
    else:
        return check_epistemic_condition(condition, functions, agent_history, obs_func)

def check_regular_condition(condition: Condition, functions: list[Function]) -> bool:
    """
    check whether the given functions is able to satisfy the given condition
    This will return the check result and the 
    """
    checking_function = get_function_with_name_and_params(
        functions, condition.condition_function_name, condition.condition_function_parameters
    )
    if checking_function is None:
        return False

    if not condition.value is None:
        if not compare_condition_values(checking_function.value, condition.value, condition.condition_operator):
            return False
    else:
        target_function = get_function_with_name_and_params(
            functions, condition.target_function_name, condition.target_function_parameters
        )
        if target_function is None:
            return False
        if not compare_condition_values(checking_function.value, target_function.value, condition.condition_operator):
            return False

    return True

def check_epistemic_condition(condition: Condition, functions: list[Function], agent_history: list[list[Function]], obs_func) -> bool:
    history = agent_history + [functions]
    for history_functions in reversed(history):
        belief_functions = get_functions_with_belief_sequence(
            history_functions, condition.belief_sequence, obs_func
        )
        check_result = util.check_regular_condition(condition, belief_functions)
        if check_result:
            return True
        find_func = get_function_with_name_and_params(
            belief_functions, condition.condition_function_name, condition.condition_function_parameters
        )
        if find_func is not None:
            return False
    return False

def get_functions_with_belief_sequence(functions: list[Function], belief_sequence: list[str], obs_func) -> list[Function]:
    if len(belief_sequence) == 0:
        return functions
    ontic_functions = copy.deepcopy(functions)
    for agent_name in belief_sequence:
        ontic_functions = obs_func.get_observable_functions(ontic_functions, agent_name)
    return ontic_functions



def get_function_with_name_and_params(functions: list[Function], name: str, params: dict[str, str]):
    """
    get the function with the given locator
    """
    for function in functions:
        if function.name == name and list(function.parameters.values()) == list(params.values()):
            return function
    return None

def is_conflict_functions(function1: Function, function2: Function) -> bool:
    """
    check whether two functions are conflict with each other
    """
    if function1.name == function2.name and function1.parameters == function2.parameters:
        if function1.value != function2.value:
            return True
    return False

def get_agent_unknown_functions(model: Model, agent_name: str) -> list[Function]:
    """
    get agent's unknown functions based on what agent knows
    """
    all_functions = model.generate_all_possible_functions()
    current_agent = model.get_agent_by_name(agent_name)
    # remove the functions that agent already knows
    all_functions = [function for function in all_functions if function not in current_agent.functions]
    unknown_functions = []
    # filter the functions that are conflict with what agent knows
    for func in all_functions:
        is_conflict = False
        for known_func in current_agent.functions:
            if util.is_conflict_functions(func, known_func):
                is_conflict = True
                break
        if not is_conflict:
            unknown_functions.append(func)
    return unknown_functions

def function_belongs_to(model: Model, function: Function) -> str:
    """
    Normally, an agent will know a function if it is clearly belongs to it.\n
    For example, agent_loc a = 1 is clearly belongs to agent a.\n
    If there are multiple agent parameters in the function, then we assume the function belongs to the first agent.\n
    For exmpale, has_secret a b means a has b's secret, then this function belongs to agent a.
    """
    for _, param_name in function.parameters.items():
        agent = model.get_agent_by_name(param_name)
        if agent is not None:
            return agent.name
    return None

def function_is_exist(functions: list[Function], func_name: str, func_params: dict[str, str]) -> int:
    count = 0
    for function in functions:
        if function.name == func_name and function.parameters == func_params:
            count += 1
    return count

def generate_virtual_model(model: Model, agent_name: str) -> Model:
    """
    Generate a virtual model based on agent_name's perspective.\n
    agent_name's functions will become model's ontic functions, agent_name's belief to other agents will become model's agents.\n
    For unknown functions, this function will find all unknwon functions, group them with their name and parameters, and randomly pick one of them in each group as the ontic function. This will also input in agent_name's functions.\n
    If the generated unknwon function is clearly belongs to an agent, then this function will add to that agent's functions.\n
    """
    unknown_functions = get_agent_unknown_functions(model, agent_name)
    # group the functions by name and parameters
    group_functions = {}
    for func in unknown_functions:
        key = func.name + func.parameters.__str__()
        if key not in group_functions:
            group_functions[key] = []
        group_functions[key].append(func)
    unknown_functions = []
    for funcs in group_functions.values():
        unknown_functions.append(funcs)
    
    virtual_model = model.copy()
    current_agent = virtual_model.get_agent_by_name(agent_name)
    choosen_pairs = []
    while True:
        # randomly pick one of the function in each group as a part of ontic function
        assume_ontic_functions = []
        this_pairs = []
        for funcs in unknown_functions:
            f = random.choice(funcs)
            i = funcs.index(f)
            assume_ontic_functions.append(f)
            this_pairs.append(i)
        # check the validity of the assume ontic functions with current ontic functions by using the rules
        if this_pairs not in choosen_pairs:
            choosen_pairs.append(this_pairs)
            # print(f"{choosen_pairs}")
            if virtual_model.rules.check_functions(current_agent.functions + assume_ontic_functions):
                break
    
    # TODO: allocate the functions and the goals to each agents
    # the functions of current agent will not change, other agent's functions will set to the observation functions based on current agent's functions
    for agent in virtual_model.agents:
        if agent.name != agent_name:
            agent.functions = virtual_model.observation_function.get_observable_functions(current_agent.functions, agent.name)
            if virtual_model.problem_type == ProblemType.COOPERATIVE:
                this_goals = []
                for goal in current_agent.goals:
                    new_goal = copy.deepcopy(goal)
                    if len(new_goal.belief_sequence) == 0:
                        new_goal.belief_sequence = [current_agent.name]
                        new_goal.ep_operator = EpistemicOperator.EQUAL
                        new_goal.ep_truth = EpistemicTruth.TRUE
                    else:
                        if new_goal.belief_sequence[-1] == agent.name:
                            new_goal.belief_sequence = new_goal.belief_sequence[:-1]
                            if len(new_goal.belief_sequence) == 0:
                                new_goal.ep_operator = EpistemicOperator.NONE
                                new_goal.ep_truth = EpistemicTruth.NONE
                    this_goals.append(new_goal)
                agent.goals = this_goals
            else:
                # TODO: this part will be added after implement the intention prediction method
                raise Exception("Intention prediction is not implemented yet")

    virtual_model.ontic_functions = copy.deepcopy(current_agent.functions)
    for func in assume_ontic_functions:
        virtual_model.ontic_functions.append(func)
        current_agent.functions.append(func)
        belongs_to = util.function_belongs_to(virtual_model, func)
        if not belongs_to is None:
            virtual_model.get_agent_by_name(belongs_to).functions.append(func)
    
    return virtual_model