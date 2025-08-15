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
        logger.error(f"result not found: \"{regex}\" in \"{string}\"")
        raise Exception(f"result not found: \"{regex}\" in \"{string}\"")
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

def is_valid_action(model: Model, action: Action, agent_name, is_ontic_checking: bool = True) -> bool:
    """
    to check whether the action is valid in realm of given functions
    """
    if action is None:
        return True
    for condition in action.pre_condition:
        if is_ontic_checking:
            if len(condition.belief_sequence) == 0:
                if not check_regular_condition(condition, model.ontic_functions):
                    return False
        else:
            if not check_condition(model, condition, agent_name):
                return False
    return True

def check_condition(model: Model, condition: Condition, agent_name):
    if len(condition.belief_sequence) == 0:
        return check_regular_condition(condition, model.get_functions_of_agent(agent_name))
    else:
        return check_epistemic_condition(condition, model, agent_name)

def get_epistemic_world(functions_sequences: list[list[Function]]) -> list[Function]:
    """
    get the epistemic world from the given function sequence\n
    this usually use when checking the epistemic condition and generating the virtual world\n
    """
    world = []
    headers = set()
    for functions in functions_sequences:
        for func in functions:
            if func.header() not in headers:
                headers.add(func.header())
                world.append(func)
    return copy.deepcopy(world)

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

def check_epistemic_condition(condition: Condition, model: Model, agent_name) -> bool:
    history = model.get_history_functions_of_agent(agent_name) + [model.get_functions_of_agent(agent_name)]
    history_beliefs: list[list[Function]] = []
    for history_functions in reversed(history):
        belief_functions = get_functions_with_belief_sequence(
            history_functions, condition.belief_sequence, model
        )
        history_beliefs.append(belief_functions)
    epistemic_world_functions = get_epistemic_world(history_beliefs)
    return util.check_regular_condition(condition, epistemic_world_functions )

def get_functions_with_belief_sequence(functions: list[Function], belief_sequence: list[str], model: Model) -> list[Function]:
    if len(belief_sequence) == 1:
        return functions
    ontic_functions = functions
    for agent_name in belief_sequence[1:]:
        ontic_functions = model.observation_function.get_observable_functions(model, ontic_functions, agent_name)
    return ontic_functions

def get_function_with_name_and_params(functions: list[Function], name: str, params: dict[str, str]):
    """
    get the function with the given locator
    """
    for function in functions:
        if function.name == name and set(function.parameters.values()) == set(params.values()):
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

def get_unknown_functions(model, functions: list[Function], agent_name: str) -> list[Function]:
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

    current_agent_functions = model.get_functions_of_agent(agent_name)
    current_agent_history_functions = model.get_history_functions_of_agent(agent_name)
    epistemic_world = get_epistemic_world(reversed(current_agent_history_functions + [current_agent_functions]))

    known_functions = epistemic_world
    unknown_functions = get_unknown_functions(model, known_functions, agent_name)
    
    # group the functions by name and parameters
    group_functions = {}
    for func in unknown_functions:
        key = func.header()
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
    for agent in virtual_model.agents:
        if agent.name != agent_name:
            if virtual_model.problem_type == ProblemType.COOPERATIVE:
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
            else:
                # 逐层过滤非有效Goal集合
                # 似乎不该写在这里？
                pass
    
    current_history = []
    new_history_functions = []
    for history in virtual_model.history:
        current_history.append(history['functions'])
        new_history = {'functions': get_epistemic_world(reversed(current_history)), 
                       'agent': history['agent'],
                       'action': history['action']}
        new_history_functions.append(new_history)
    virtual_model.history = new_history_functions
    
    virtual_model.ontic_functions = copy.deepcopy(known_functions)
    all_virtual_models = []
    for comb in valid_combs:
        new_model = virtual_model.copy()
        new_model.ontic_functions.extend(comb)
        if new_model.problem_type == ProblemType.NEUTRAL:
            # # 基于belief_other_goals过滤all_possible_goals中的元素
            # remain_goals = current_agent.all_possible_goals[:]
            # for name, poss_goals in current_agent.belief_other_goals.items():
            #     checking_functions = poss_goals
            #     own_goal_to_other_goal = set()
            #     for goal in current_agent.own_goals:
            #         new_goal = copy.deepcopy(goal)
            #         new_goal.belief_sequence[0] = name
            #         new_goal.belief_sequence = remove_continue_duplicates(new_goal.belief_sequence)
            #         own_goal_to_other_goal.add(new_goal)
                
            #     # 只要belief goals中任意goals集合为possible goals集合中goal set的子集，便将goal set添加到valid goals中
            #     valid_goals_belief = []
            #     valid_goals_own = []
            #     for goal_set in remain_goals:
            #         for check_func in checking_functions:
            #             if check_func.issubset(set(goal_set[name])):
            #                 valid_goals_belief.append(goal_set)
            #         if own_goal_to_other_goal.issubset(set(goal_set[name])):
            #             valid_goals_own.append(goal_set)
            #     if len(valid_goals_belief) > 0:
            #         remain_goals = valid_goals_belief
            #     else:
            #         remain_goals = valid_goals_own
            remain_goals = current_agent.all_possible_goals
                

                # output = f"Agent {agent_name} belief goals:\n"
                # for goal_set in remain_goals:
                #     for name, goals in goal_set.items():
                #         output += f"{name}:\n"
                #         for goal in goals:
                #             output += f"{goal}\n"
                #     output += f"{util.SMALL_DIVIDER}"
                # new_model.logger.debug(output)

            
            # 用remain_goals中的内容创建virtual models
            for goal_set in remain_goals:
                new_model2 = new_model.copy()
                for agent in new_model2.agents:
                    if agent.name != agent_name:
                        agent.own_goals = goal_set[agent.name]
                all_virtual_models.append(new_model2)
        else:
            all_virtual_models.append(new_model)
    
    if len(all_virtual_models) <= 0:
        if len(unknown_functions) == 0:
            all_virtual_models.append(virtual_model)
        else:
            print(len(all_virtual_models))
            kf = ""
            for f in known_functions:
                kf += f"{f}\n"
            print(f"known functions:\n{kf}\n")
            kf = ""
            for f in unknown_functions:
                kf += f"{f}\n"
            print(f"unknown functions:\n{kf}\n")
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
def check_bfs(virtual_model: Model) -> bool:
    heap: list[BFSNode] = []
    current_agent_index = 0
    count_agent = len(virtual_model.agents)
    heapq.heappush(heap, BFSNode(current_agent_index, [], virtual_model, 0))
    existed_epistemic_world = {agt.name: set() for agt in virtual_model.agents}
    find_solution_depth = -1
    while heap:
        node = heapq.heappop(heap)
        if (find_solution_depth != -1 and len(node.actions) > find_solution_depth
            or len(node.actions) > 12):
            break

        if node.model.full_goal_complete():
            return True
        current_agent = node.model.agents[node.current_index]
        successors = node.model.get_agent_successors(current_agent.name)
        successors = [succ for succ in successors if util.is_valid_action(node.model, succ, current_agent.name)]
        for succ in successors:
            next_model = node.model.copy()
            next_model.move(current_agent.name, succ)
            # 过滤机制
            ep_funcs = []
            for agt in next_model.agents:
                his_ep_funcs = next_model.get_history_functions_of_agent(agt.name)
                cur_ep_funcs = next_model.get_functions_of_agent(agt.name)
                ep_funcs.append(frozenset(util.get_epistemic_world(reversed(his_ep_funcs + [cur_ep_funcs]))))
            if frozenset(ep_funcs) in existed_epistemic_world[current_agent.name]:
                continue
            existed_epistemic_world[current_agent.name].add(frozenset(ep_funcs))

            heapq.heappush(heap, 
                        BFSNode((node.current_index + 1) % count_agent,
                                    node.actions + [succ],
                                    next_model,
                                    node.priority + 1))
    return False

def simulate_a_round(model: Model, current_agent: str):
    end_models = []

    heap: list[BFSNode] = []
    end_index = model.get_agent_index_by_name(current_agent)
    count_agent = len(model.agents)
    cur_index = (end_index + 1) % count_agent
    heapq.heappush(heap, BFSNode(cur_index, [], model, 0))
    existed_epistemic_world = {agt.name: set() for agt in model.agents}
    while heap:
        node = heapq.heappop(heap)

        if (node.current_index + 1 == end_index and len(node.actions) != 0) or node.model.full_goal_complete():
            end_models.append(node.model)
            continue
        current_agent = node.model.agents[node.current_index]
        successors = node.model.get_agent_successors(current_agent.name)
        successors = [succ for succ in successors if util.is_valid_action(node.model, succ, current_agent.name)]
        for succ in successors:
            next_model = node.model.copy()
            next_model.move(current_agent.name, succ)

            # 过滤机制
            ep_funcs = []
            for agt in next_model.agents:
                his_ep_funcs = next_model.get_history_functions_of_agent(agt.name)
                cur_ep_funcs = next_model.get_functions_of_agent(agt.name)
                ep_funcs.append(frozenset(util.get_epistemic_world(reversed(his_ep_funcs + [cur_ep_funcs]))))
            if frozenset(ep_funcs) in existed_epistemic_world[current_agent.name]:
                continue
            existed_epistemic_world[current_agent.name].add(frozenset(ep_funcs))
            
            heapq.heappush(heap, 
                        BFSNode((node.current_index + 1) % count_agent,
                                    node.actions + [succ],
                                    next_model,
                                    node.priority + 1))
    return end_models

class BFSNode:
    def __init__(self, current_index, action, model, priority):
        self.current_index: int = current_index
        self.actions: list[Action] = action
        self.model: Model = model
        self.priority: int = priority
    
    def __lt__(self, other):
        return self.priority < other.priority