import logging
import inspect
import re
import importlib.util
from itertools import product
from pathlib import Path
from time import perf_counter
from functools import wraps

BIG_DIVIDER = "=================\n"
MEDIUM_DIVIDER = "*****************\n"
SMALL_DIVIDER = "-----------------\n"

MODEL_FOLDER_PATH = "models/"
OBS_FUNC_FOLER_PATH = "observation_functions/"
STRATEGY_FOLDER_PATH = "policy_strategies/"
RULES_FOLDER_PATH = "rules/"
INIT_FILE_NAME = "init.envpddl"
AGENT_FILE_NAME = ".agtpddl"
INIT_TEMPLATE_PATH = "models/init_template.txt"
AGENT_TEMPLATE_PATH = "models/agt_template.txt"

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

def record_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} 耗时: {end - start:.6f}秒")
        return result
    return wrapper

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
        logger.error(f"result not found: {regex} in {string}")
        raise Exception(f"result not found: {regex} in {string}")
    return result

def regex_match(regex, string, logger=None):
    result = re.match(regex, string, re.M)
    if logger and not result :
        logger.error(f"result not found: \"{regex}\"")
        raise Exception(f"result not found: \"{regex}\"")
    return True if result else False

from epistemic_handler.file_parser import *
from epistemic_handler.epistemic_class import *

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
    if not isinstance(a, type(b)) or (isinstance(a, str) and strategy not in [ConditionOperator.EQUAL, ConditionOperator.NOT_EQUAL]):
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

def is_valid_action(model: Model, action: Action) -> bool:
    """
    to check whether the action is valid
    """
    if action is None:
        return True
    for condition in action.pre_condition:
        if not check_condition(model, condition):
            return False
    return True

def check_condition(model: Model, condition: Condition):
    epistemic_world_functions = get_epistemic_world(model, condition.belief_sequence)
    return check_regular_condition(condition, epistemic_world_functions)
        

def get_unfiltered_st(world_seq: list[list[Function]]) -> list[Function]:
    """
    get the epistemic world from the given function sequence\n
    this usually use when checking the epistemic condition and generating the virtual world\n
    """
    if len(world_seq) == 0:
        return []
    
    world = []
    headers = set()
    for functions in reversed(world_seq):
        for func in functions:
            if func.header_id not in headers:
                headers.add(func.header_id)
                world.append(func)
    return world

def get_epistemic_world(model: Model, belief_sequence: list[str], history_functions=[]) -> list[Function]:
    """
    if belief_sequence = [a,b,c], history = [S0, S1, ..., Sn]
    output: st' = st'' / ( Oc(st'') / Oc(st) )
    st = fb(fa(St))
    st'' = fc(fb(fa(St)))
    """
    if len(history_functions) == 0:
        history_functions = model.get_history_functions()
    if len(history_functions) == 0:
        return []
    if len(belief_sequence) == 0:
        return history_functions[-1]

    # st''
    history_beliefs = [get_functions_with_belief_sequence(functions, belief_sequence, model) for functions in history_functions]
    st2 = get_unfiltered_st(history_beliefs)

    # st
    st = get_epistemic_world(model, belief_sequence[:-1], history_functions)

    # Oi(st'')
    Oi_st2 = set(model.observation_function.get_observable_functions(model, st2, belief_sequence[-1]))
    
    # Oi(st)
    Oi_st = set(model.observation_function.get_observable_functions(model, st, belief_sequence[-1]))

    return list(set(st2).difference(Oi_st2.difference(Oi_st)))


def check_regular_condition(condition: Condition, functions: list[Function]) -> bool:
    """
    check whether the given functions is able to satisfy the given condition
    This will return the check result and the 
    """
    checking_function = get_function_with_name_and_params(
        functions, condition.condition_function_name, condition.condition_function_parameters
    )
    # solve the situation when it is an epistemic condition with an ep.none operator in it
    # if it is ep.none, then we only need to check whther the checking_function is exist or not depends on the epistemic operator
    if condition.ep_truth == EpistemicTruth.UNKNOWN:
        return checking_function is None if condition.ep_operator == EpistemicOperator.EQUAL else checking_function is not None
    
    if checking_function is None:
        return False

    if condition.value is not None:
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

def get_functions_with_belief_sequence(functions: list[Function], belief_sequence: list[str], model: Model) -> list[Function]:
    if len(belief_sequence) == 0:
        return functions
    ontic_functions = functions
    for agent_name in belief_sequence:
        ontic_functions = model.observation_function.get_observable_functions(model, ontic_functions, agent_name)
    return ontic_functions

def get_function_with_name_and_params(functions: list[Function], name: str, params: dict[str, str]):
    """
    get the function with the given locator
    """
    for function in functions:
        if function.name == name and frozenset(function.parameters.values()) == frozenset(params.values()):
            return function
    return None

def is_conflict_functions(function1: Function, function2: Function) -> bool:
    """
    check whether two functions are conflict with each other
    """
    return function1.id != function2.id

def get_unknown_functions(model: Model, functions: list[Function], agent_name: str) -> list[Function]:
    """
    get agent's unknown functions based on what agent knows
    """
    all_functions = model.generate_all_possible_functions()
    # remove the functions that agent already knows

    all_functions = [function for function in all_functions if function not in functions]
    unknown_functions = []
    # filter the functions that are conflict with what agent knows
    for func in all_functions:
        is_conflict = False
        for known_func in functions:
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

    known_functions = get_epistemic_world(model, [agent_name])
    unknown_functions = get_unknown_functions(model, known_functions, agent_name)
    
    # group the functions by name and parameters
    group_functions = {}
    for func in unknown_functions:
        key = func.header_id
        if key not in group_functions:
            group_functions[key] = []
        group_functions[key].append(func)
    unknown_functions = []
    for funcs in group_functions.values():
        unknown_functions.append(funcs)

    all_combs = product(*group_functions.values())
    valid_combs = []
    for comb in all_combs:
        if model.rules.check_functions(known_functions + list(comb)):
            valid_combs.append(comb)
    
    virtual_model = model.copy()
    current_agent = virtual_model.get_agent_by_name(agent_name)
    # the functions of current agent will not change, other agent's functions will set to the observation functions based on current agent's functions
    if virtual_model.problem_name == ProblemType.COOPERATIVE:
        for agent in virtual_model.agents:
            if agent.name != agent_name:
                if current_agent.other_goals[agent.name]:
                    agent.own_goals = current_agent.other_goals[agent.name]
                else:
                    # 转化goal
                    goals = []
                    for goal in current_agent.own_goals:
                        new_goal = copy.deepcopy(goal)
                        new_goal.belief_sequence[0] = agent.name
                        new_goal.belief_sequence = remove_continue_duplicates(new_goal.belief_sequence)
                        goals.append(new_goal)
                    agent.own_goals = goals
    
    # update the model history to the history based on current_agent's perspective 
    current_history = []
    new_history_functions = []
    for history in virtual_model.history:
        current_history.append(history['functions'])
        new_history = {'functions': get_epistemic_world(virtual_model, [agent_name], history_functions=current_history), 
                       'agent': history['agent'],
                       'action': history['action'],
                       'signal': history['signal']}
        new_history_functions.append(new_history)
    virtual_model.history = new_history_functions
    
    virtual_model.ontic_functions = known_functions
    all_virtual_models = []
    for comb in valid_combs:
        new_model = virtual_model.copy()
        new_model.ontic_functions.extend(comb)
        if new_model.problem_type == ProblemType.NEUTRAL:
            for goal_set in current_agent.all_possible_goals:
                new_model2 = new_model.copy()
                for agent in new_model2.agents:
                    agent.own_goals = goal_set[agent.name]
                all_virtual_models.append(new_model2)
        else:
            all_virtual_models.append(new_model)
    
    if len(all_virtual_models) <= 0:
        if len(unknown_functions) == 0:
            all_virtual_models.append(virtual_model)
        else:
            model.logger.debug("unable to generate the virtual world")
            if model.problem_type == ProblemType.NEUTRAL:
                model.logger.debug(f"agent belief goals num: {len(current_agent.all_possible_goals)}")
            print("unable to generate the virtual world")
            model.logger.debug(f"valid combs num: {len(valid_combs)}, valid possible goals num: {len(current_agent.all_possible_goals)}")
            kf = ""
            for f in known_functions:
                kf += f"{f}\n"
            model.logger.debug(f"known functions:\n{kf}")
            kf = ""
            for f in unknown_functions:
                kf += f"{f}\n"
            model.logger.debug(f"unknown functions:\n{kf}")
            exit(0)
    return all_virtual_models

def remove_continue_duplicates(lst):
    if not lst:
        return []
    new_list = [lst[0]]
    for ele in lst:
        if ele != new_list[-1]:
            new_list.append(ele)
    return new_list

import heapq
def check_bfs(virtual_model: Model, max_action_length=-1) -> int:
    heap: list[BFSNode] = []
    heapq.heappush(heap, BFSNode(1, [], virtual_model, 0))
    existed_epistemic_world = set()
    while heap:
        node = heapq.heappop(heap)
        if node.model.full_goal_complete():
            return len(node.actions)
        
        if max_action_length > 0 and len(node.actions) == max_action_length:
            break
        successors = {}
        for agent in node.model.agents:
            successors[agent.name] = node.model.get_agent_successors(agent.name)
        for name, succs in successors.items():
            for succ in succs:
                next_model = node.model.copy()
                next_model.move(name, succ)
                # 过滤机制
                observe_funcs = frozenset([frozenset([agt.name] + get_epistemic_world(next_model, [agt.name])) for agt in next_model.agents])
                if observe_funcs in existed_epistemic_world:
                    continue
                existed_epistemic_world.add(observe_funcs)

                heapq.heappush(heap, 
                            BFSNode(1,
                                        node.actions + [succ],
                                        next_model,
                                        node.priority + 1))
    
    return -1

class BFSNode:
    def __init__(self, current_index, action, model, priority):
        self.current_index: int = current_index
        self.actions: list[Action] = action[:]
        self.model: Model = model
        self.priority: int = priority
    
    def __lt__(self, other):
        return self.priority < other.priority

def load_action_sequence(path: str, model: Model, logger) -> list[Action]:
    path = f"models/{path}"
    with open(path, 'r') as f:
        content = f.read()
        logger.info(f"Complete reading action sequence file \"{path}\"\n{content}")
    lines = content.split("\n")

    result = []
    # each line is an action, such as a: action param1 param2 ...
    from epistemic_handler.epistemic_class import Action
    for line in lines:
        keys = line.split(" ")
        move_agent = keys[0][:-1]
        action_name = keys[1]
        parameters = keys[2:]
        if action_name == "stay":
            new_action = Action.stay_action(move_agent)
            result.append([move_agent, new_action])
            continue
        for action_schema in model.action_schemas:
            if action_schema.name == action_name:
                params = dict(zip(action_schema.require_parameters.keys(), parameters))
                new_action = Action.create_action(action_schema, params)
                result.append([move_agent, new_action])
                break
    
    output = ""
    for action in result:
        output += f"{action[0]}: {action[1].header()}\n"
    logger.info(f"Complete parsing the actions:\n{output}")
    print(f"Complete parsing the actions:\n{output}")
    return result

class FinalFunctions:
    """
    This is a class to store all possible functions and maintain them during the program running.
    Hope this will fix the issues of cpu, time pressure due to tons of deepcopy usage.
    The update of a function will now delete the function pointer from the original list and add a target function pointer to the list.
    Well, basically, this class maintains all possible functions.
    """
    def __init__(self):
        """
        the structure of the dict is:
        all:
            function_name1:
                frozenset(func.parameters.values()):
                    func.value:
                        func
                    ...
            function_name2:
                ...
        """
        
        self.all: dict[str, dict[str, dict[str, Function]]] = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    
    def add_function(self, function: Function) -> None:
        # get the parameters of function
        # to make sure no order problem will happen during the "get" method, we should use frozenset
        params = f"{list(function.parameters.values())}"
        self.all[function.name][params][str(function.value)] = function
    
    def get_function(self, function_name: str, parameters: dict[str, str], value: str):
        params = f"{list(parameters.values())}"
        if self.all[function_name][params][str(value)] is None:
            raise Exception(f"Function {function_name} with parameters {parameters} and value {value} is not found.")
        return self.all[function_name][params][str(value)]
    
    def flatten(self) -> list[Function]:
        return [
            v3
            for v1 in self.all.values()
            for v2 in v1.values()
            for v3 in v2.values()
        ]

    def __str__(self):
        output = ""
        for function_name, function_dict in self.all.items():
            output += f"Function {function_name}:\n"
            for params, value_dict in function_dict.items():
                output += f"Parameters {params}:\n"
                for value, function in value_dict.items():
                    output += f"Value {value}: {function}\n"
        return output
    
    def __repr__(self):
        return self.__str__()