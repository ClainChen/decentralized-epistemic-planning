from enum import Enum
import logging
import util
from itertools import combinations, product, permutations
import copy
from dataclasses import dataclass, field

MODEL_LOGGER_LEVEL = logging.DEBUG
class ProblemType(Enum):
    COOPERATIVE = 1
    NEUTRAL = 2

class EpistemicOperator(Enum):
    EQUAL = 1
    NOT_EQUAL = 2
    NONE = 0

    def __str__(self):
        return f"\"{self.name}\""

    def __repr__(self):
        return self.__str__()

class EpistemicTruth(Enum):
    TRUE = 1
    FALSE = 2
    UNKNOWN = 3
    NONE = 0

    def __str__(self):
        return f"\"{self.name}\""

    def __repr__(self):
        return self.__str__()

class ConditionOperator(Enum):
    EQUAL = 1
    NOT_EQUAL = 2
    GREATER = 3
    GREATER_EQUAL = 4
    LESS = 5
    LESS_EQUAL = 6
    NONE = 0

    def __str__(self):
        return f"\"{self.name}\""

    def __repr__(self):
        return self.__str__()

class EffectType(Enum):
    ASSIGN = 1
    INCREASE = 2
    DECREASE = 3
    NONE = 0

    def __str__(self):
        return f"\"{self.name}\""

    def __repr__(self):
        return self.__str__()

class ValueType(Enum):
    INTEGER = 1
    ENUMERATE = 2
    NONE = 3

    def __str__(self):
        return f"\"{self.name}\""

    def __repr__(self):
        return self.__str__()

PROBLEM_TYPE_MAPS = {
    "cooperative": ProblemType.COOPERATIVE,
    "neutral": ProblemType.NEUTRAL
}

EPISTEMIC_OPERATOR_MAPS = {
    "=": EpistemicOperator.EQUAL,
    "!=": EpistemicOperator.NOT_EQUAL
}

EPISTEMIC_TRUTH_MAPS = {
    "true": EpistemicTruth.TRUE,
    "false": EpistemicTruth.FALSE,
    "unknown": EpistemicTruth.UNKNOWN
}

CONDITION_OPERATOR_MAPS = {
    "=": ConditionOperator.EQUAL,
    "!=": ConditionOperator.NOT_EQUAL,
    ">": ConditionOperator.GREATER,
    "<": ConditionOperator.LESS,
    ">=": ConditionOperator.GREATER_EQUAL,
    "<=": ConditionOperator.LESS_EQUAL
}

EFFECT_TYPE_MAPS = {
    "assign": EffectType.ASSIGN,
    "increase": EffectType.INCREASE,
    "decrease": EffectType.DECREASE,
}
@dataclass
class Entity:
    name: str = None
    type: str = None
    
class FunctionSchema:
    def __init__(self):
        self.name: str = None
        self.range: tuple[int, int] | list = None
        self.type: ValueType = ValueType.NONE
        self.require_parameters: dict[str, str] = {}
    
    def __str__(self):
        return f"FunctionSchema(name: {self.name}, range: {self.range}, type: {self.type}, require_parameters: {self.require_parameters})"
    
    def __repr__(self):
        return self.__str__()

@dataclass
class Function:
    """
    Use as a state or an epistemic state, this will store the real value
    """
    name: str = None
    range: tuple[int, int] | list = None
    type: ValueType = ValueType.NONE
    parameters: dict[str, str] = field(default_factory=dict)
    _value: int | str = None

    @property
    def value(self) -> int | str:
        return self._value
    
    @value.setter
    def value(self, value):
        if hasattr(self, 'type'):
            if self.type == ValueType.INTEGER:
                self._value = int(value)
            else:
                self._value = value
        else:
            raise ValueError("Cannot set value before type is set")

    def compare(self, other: 'Function'):
        return self.name == other.name and self.parameters == other.parameters

    def __str__(self):
        params = [f"{key} - {value}" for key, value in self.parameters.items()]
        return f"Function({self.name} {params} = {self.value})"

    def __repr__(self):
        return self.__str__()

class FunctionLocator:
    """
    Use in the condition phase, this will not include the value, the value of this function is always in the upper layer: Condition
    """
    def __init__(self):
        self.name: str = None
        self.range: tuple[int, int] | list = None
        self.type: ValueType = ValueType.NONE
        self.parameters: dict[str, str] = []
    
    @classmethod
    def proper_build(cls, function_schema: FunctionSchema, parameters: dict[str, Entity]) -> 'FunctionLocator':
        locator = cls()
        locator.name = function_schema.name
        locator.range = function_schema.range
        locator.type = function_schema.type
        locator.parameters = {key: parameters[key] for key in function_schema.require_parameters.keys()}
        return locator

    def __str__(self):
        return f"FunctionLocator(name: {self.name}, range: {self.range}, type: {self.type}, parameters: {self.parameters})"

class ConditionSchema:
    def __init__(self):
        self.ep_operator: EpistemicOperator = EpistemicOperator.NONE
        self.belief_sequence: list[str] = []
        self.ep_truth: EpistemicTruth = EpistemicTruth.NONE
        self.condition_operator: ConditionOperator = ConditionOperator.NONE
        self.condition_function_schema: FunctionSchema = None
        self.value: int | str = None
        self.target_function_schema: FunctionSchema = None
    
    def __str__(self):
        result = f"Condition(ep_operator: {self.ep_operator}, belief_sequence: {self.belief_sequence}, ep_truth: {self.ep_truth}, condition_operator: {self.condition_operator}, condition_function_schema: {self.condition_function_schema}, value: {self.value} / target_function_schema: {self.target_function_schema})\n"
        return result

    def __repr__(self):
        return self.__str__()

class Condition:
    def __init__(self, condition_schema: ConditionSchema, parameters: dict[str, str]):
        self.ep_operator: EpistemicOperator = condition_schema.ep_operator
        self.belief_sequence: list[str] = condition_schema.belief_sequence
        self.ep_truth: EpistemicTruth = condition_schema.ep_truth
        self.condition_operator: ConditionOperator = condition_schema.condition_operator
        self.condition_function_locator: FunctionLocator = FunctionLocator.proper_build(condition_schema.condition_function_schema, parameters)
        self.value: int | str = condition_schema.value
        if condition_schema.target_function_schema is not None:
            self.target_function_locator: FunctionLocator = FunctionLocator.proper_build(condition_schema.target_function_schema, parameters)
        else:
            self.target_function_locator = None

    def __str__(self):
        result = "Condition:\n"
        result += f"ep_operator: {self.ep_operator}, belief_sequence: {self.belief_sequence}, ep_truth: {self.ep_truth}\n"
        result += f"condition_operator: {self.condition_operator}, condition_function_locator: {self.condition_function_locator}\n"
        result += f"value: {self.value} / target_function_locator: {self.target_function_locator}"
        return result

    def __repr__(self):
        return self.__str__()

class EffectSchema:
    def __init__(self):
        self.effect_type: EffectType = EffectType.NONE
        self.effect_function_schema: FunctionSchema = None
        self.value: int | str = None
        self.target_function_schema: FunctionSchema = None

    def __str__(self):
        result = f"EffectSchema(effect_type: {self.effect_type}, effect_function_schema: {self.effect_function_schema}, value: {self.value})"
        return result
    
    def __repr__(self):
        return self.__str__()

class Effect:
    def __init__(self, effect_schema: EffectSchema, parameters: dict[str, Entity]):
        self.effect_type: EffectType = effect_schema.effect_type
        self.effect_function_locator: FunctionLocator = FunctionLocator.proper_build(effect_schema.effect_function_schema, parameters)
        self.value: int | str = effect_schema.value
        if effect_schema.target_function_schema is not None:
            self.target_function_locator: FunctionLocator = FunctionLocator.proper_build(effect_schema.target_function_schema, parameters)
        else:
            self.target_function_locator = None
    def __str__(self):
        result = "EffectSchema:\n"
        result += f"effect_type: {self.effect_type}\n"
        result += f"effect_function_locator: {self.effect_function_locator}\n"
        result += f"value: {self.value}"
        result += f"target_function_locator: {self.target_function_locator}\n"
        return result

class ActionSchema:
    def __init__(self):
        self.name: str = None
        self.require_parameters: dict[str, str] = {}
        self.pre_condition_schemas: list[ConditionSchema] = []
        self.effect_schemas: list[EffectSchema] = []
    
    def get_type_count(self) -> dict[str, int]:
        '''
        Get number of types
        '''
        counts = {}
        types = set(self.require_parameters.values())
        for type in types:
            counts[type] = len([t for t in self.require_parameters.values() if t == type])
        return counts
    
    def __str__(self):
        result = f"Action Schema: {self.name}\n"
        # result += util.MEDIUM_DIVIDER
        result += f"Parameters: {self.require_parameters}\n"
        # result += util.MEDIUM_DIVIDER
        result += f"Precondition Schemas:\n"
        for pre_condition_schema in self.pre_condition_schemas:
            # result += util.SMALL_DIVIDER
            result += f"{pre_condition_schema}\n"
        # result += util.MEDIUM_DIVIDER
        result += f"Effect Schemas:\n"
        for effect_schema in self.effect_schemas:
            # result += util.SMALL_DIVIDER
            result += f"{effect_schema}\n"
        return result

    def __repr__(self):
        return self.__str__()

class Action:
    def __init__(self, action_schema: ActionSchema, parameters: dict[str, str]):
        self.name = action_schema.name
        self.parameters: dict[str, str] = parameters
        self.pre_condition: list[Condition] = []
        for condition_schema in action_schema.pre_condition_schemas:
            self.pre_condition.append(Condition(condition_schema, parameters))
        self.effect: list[Effect] = []
        for effect_schema in action_schema.effect_schemas:
            self.effect.append(Effect(effect_schema, parameters))

    def __str__(self):
        result = f"Action:\n"
        result += f"name: {self.name}\nparameters: {self.parameters}\n"
        result += "Pre Conditions:\n"
        for condition in self.pre_condition:
            result += f"{condition}\n"
        result += "Effects:\n"
        for effect in self.effect:
            result += f"{effect}\n"
        return result

    def __repr__(self):
        return self.__str__()

class Goal:
    def __init__(self):
        self.ep_operator: EpistemicOperator = EpistemicOperator.NONE
        self.belief_sequence: list[str] = []
        self.ep_truth: EpistemicTruth = EpistemicTruth.NONE
        self.condition_operator: ConditionOperator = ConditionOperator.NONE
        self.goal_function_name: str = None
        self.goal_function_parameters: list[str] = []
        self.value: int | str = None
        self.target_function_name: str = None
        self.target_function_parameters: list[str] = []

    def __str__(self):
        result = f"Goal(ep_operator: {self.ep_operator}, belief_sequence: {self.belief_sequence}, ep_truth: {self.ep_truth}, condition_operator: {self.condition_operator}, goal_function_name: {self.goal_function_name}, goal_function_parameters: {self.goal_function_parameters}, value: {self.value} / target_function_name: {self.target_function_name}, target_function_parameters: {self.target_function_parameters})"
        return result

    def __repr__(self):
        return self.__str__()

class Agent:
    def __init__(self):
        self.name: str = None
        self.functions: list[Function] = []
        self.goals: list[Goal] = []
        self.history_functions: list[list[Function]] = []
        self.belief_to_other_agents: list[Agent] = []

    def copy(self):
        new_agent = Agent()
        new_agent.name = self.name
        new_agent.goals = self.goals
        new_agent.functions = copy.deepcopy(self.functions)
        new_agent.history_functions = copy.deepcopy(self.history_functions)
        for agent in self.belief_to_other_agents:
            new_agent.belief_to_other_agents.append(agent.copy())
        return new_agent
    
    def get_belief_of_agent(self, agent_name: str):
        """
        get the agent from belief_to_other_agents based on the given agent_name
        """
        return next((agent for agent in self.belief_to_other_agents if agent.name == agent_name), None)

    def update_functions(self, functions: list[Function]):
        self.history_functions.append(copy.deepcopy(self.functions))

        for function in functions:
            updating_function = None
            for agent_function in self.functions:
                if function.compare(agent_function):
                    updating_function = agent_function
                    break
            if updating_function is None:
                self.functions.append(copy.deepcopy(function))
            else:
                agent_function.value = function.value

    def is_complete(self):
        # TODO: #4 这里要加epistemic条件的判断
        for goal in self.goals:
            function = self.get_function_with_name_and_params(goal.goal_function_name, goal.goal_function_parameters)
            if function is None:
                return False
            else:
                if goal.value is not None:
                    if not util.compare_condition_values(goal.value, function.value, goal.condition_operator):
                        return False
                else:
                    target_function = self.get_function_with_name_and_params(goal.target_function_name, goal.target_function_parameters)
                    if target_function is None or not util.compare_condition_values(goal.value, target_function.value, goal.condition_operator):
                        return False
        return True

    def get_function_with_name_and_params(self, name: str, params: list[str]):
        for function in self.functions:
            if function.name == name and list(function.parameters.values()) == params:
                return function
        return None

    def __str__(self):
        result = f"Agent: {self.name}\n"
        result += f"Functions:\n"
        for function in self.functions:
            result += f"{function}\n"
        result += f"Goals:\n"
        for goal in self.goals:
            result += util.SMALL_DIVIDER
            result += f"{goal}\n"
        result += util.MEDIUM_DIVIDER
        result += f"History Functions:\n"
        round = 1
        for functions in self.history_functions:
            result += util.SMALL_DIVIDER
            result += f"{round}:\n"
            for function in functions:
                result += f"{function}\n"
            round += 1
        result += util.MEDIUM_DIVIDER
        result += f"Belief to other agents:\n"
        for agent in self.belief_to_other_agents:
            result += util.SMALL_DIVIDER
            result += f"{agent}:\n"
        return result

    def __repr__(self):
        return self.__str__()

class Model:
    def __init__(self):
        from abstracts import AbstractObservationFunction, AbstractPolicyStrategy
        self.logger = None
        self.observation_function: AbstractObservationFunction = None
        self.strategy: AbstractPolicyStrategy = None
        self.problem_type: ProblemType = None

        self.domain_name: str = None
        self.problem_name: str = None
        self.function_schemas: list[FunctionSchema] = []
        self.ontic_functions: list[Function] = []
        self.entities: list[Entity] = []
        self.action_schemas: list[ActionSchema] = []
        self.agents: list[Agent] = []
    
    def init(self, handler, problem_type, observation_function_path, policy_strategy_path):
        from abstracts import AbstractObservationFunction, AbstractPolicyStrategy
        self.logger = util.setup_logger(__name__, handler, logger_level=MODEL_LOGGER_LEVEL)
        
        ObsFunc = util.load_observation_function(observation_function_path, self.logger)
        self.observation_function: AbstractObservationFunction = ObsFunc(handler)
        self.logger.info(f"Loaded observation function: {ObsFunc.__name__}")

        Strategy = util.load_policy_strategy(policy_strategy_path, self.logger)
        self.strategy: AbstractPolicyStrategy = Strategy(handler)
        self.logger.info(f"Loaded policy strategy: {Strategy.__name__}")

        self.problem_type: ProblemType = PROBLEM_TYPE_MAPS[problem_type]
    
    def get_function_schema_by_name(self, name: str) -> FunctionSchema:
        return copy.deepcopy(next((function_schema for function_schema in self.function_schemas if function_schema.name == name), None))
    
    def get_all_entity_name_by_type(self, type: str) -> list[str]:
        """
        get all entity name by type
        """
        return [entity.name for entity in self.entities if entity.type == type]

    def get_agent_by_name(self, name: str) -> Agent:
        """
        get agent by name
        """
        return next((agent for agent in self.agents if agent.name == name), None)

    def get_agent_index_by_name(self, name: str) -> int:
        """
        get agent index by name
        """
        return self.agents.index(self.get_agent_by_name(name))

    def simulate(self):
        """
        Simulate the model until all agents have reached a terminal state
        """
        agent_index = 0
        agent_count = len(self.agents)
        while not self.full_goal_complete():
            self.agent_move(self.agents[agent_index].name)
            agent_index = (agent_index + 1) % agent_count

    def agent_move(self, agent_name: str):
        """
        The processes that happens when an agent wants to move
        """
        self.observe_and_update_agent(agent_name)
        action = self.strategy.get_policy(self, agent_name)
        if action is not None and util.is_valid_action(self.ontic_functions, action):
            self.do_action(agent_name, action)
            print(f"{agent_name} takes action: {action.name}")
        else:
            self.do_action(agent_name, None)
            print(f"{agent_name} takes action: stay")
        # TODO: #5 需要思考一下如何进行intention prediction

    def observe_and_update_agent(self, agent_name: str):
        agent = self.get_agent_by_name(agent_name)
        observe_functions = self.observation_function.get_observable_functions(model=self, agent_name=agent_name)
        # update agent's functions
        for agt_name, functions in observe_functions.items():
            if agt_name == agent.name:
                agent.update_functions(functions)
            else:
                belief_of_agent = agent.get_belief_of_agent(agt_name)
                belief_of_agent.update_functions(functions)

    def do_action(self, agent_name: str, action: Action) -> bool:
        if action is None:
            return True
        agent = self.get_agent_by_name(agent_name)
        
        if not util.is_valid_action(self.ontic_functions, action):
            return False
        for effect in action.effect:
            # update agent's function
            self._update_functions(agent.functions, effect)

            # update ontic functions
            self._update_functions(self.ontic_functions, effect)
        return True

    def _update_functions(self, functions: list[Function], effect: Effect):
        function = util.get_function_with_locator(functions, effect.effect_function_locator)
        if function is not None:
            if effect.value is not None:
                function.value = util.compare_effect_values(function.value, effect.value, effect.   effect_type)
            else:
                target_function = util.get_function_with_locator(self.ontic_functions, effect.   target_function_locator)
                function.value = util.compare_effect_values(function.value, target_function.value, effect.  effect_type)
        else:
            if effect.effect_type != EffectType.ASSIGN:
                self.logger.error(f"Trying to change a function that not exist in agent's functions or ontic world functions")
                raise ValueError("Trying to change a function that not exist in agent's functions or ontic world functions")
            if effect.value is not None:
                function = Function()
                function.name = effect.effect_function_locator.name
                function.range = effect.effect_function_locator.range
                function.type = effect.effect_function_locator.type
                function.parameters = effect.effect_function_locator.parameters
                function.value = effect.value
                functions.append(function)
            else:
                target_function = util.get_function_with_locator(self.ontic_functions, effect.  target_function_locator)
                if target_function is None:
                    self.logger.error(f"Target function {effect.target_function_locator.name} not found in effect phase")
                    raise ValueError(f"Target function {effect.target_function_locator.name} not found in effect phase")

                function = Function()
                function.name = effect.effect_function_locator.name
                function.range = effect.effect_function_locator.range
                function.type = effect.effect_function_locator.type
                function.parameters = effect.effect_function_locator.parameters
                function.value = target_function.value
                functions.append(function)
        assert util.check_in_range(function)

    def get_agent_successors(self, agent_name: str) -> list[Action]:
        if self.agent_goal_complete(agent_name):
            return []

        agent = self.get_agent_by_name(agent_name)
        candidates = []
        for action_schema in self.action_schemas:
            poss_params = []
            for param_name, param_type in action_schema.require_parameters.items():
                if param_name == '?self':
                    poss_params.append([agent.name])
                else:
                    poss_params.append(self.get_all_entity_name_by_type(param_type))
            poss_params = [comb for comb in product(*poss_params) if not util.check_duplication(comb)]
            result_params = [dict(zip(action_schema.require_parameters.keys(), sub_comb)) for sub_comb in poss_params]
            for param in result_params:
                successor = Action(action_schema, param)
                candidates.append(successor)
        result = []
        for action in candidates:
            if util.is_valid_action(agent.functions, action):
                result.append(action)
        return result

    def agent_goal_complete(self, agent_name: str):
        agent = self.get_agent_by_name(agent_name)
        return agent.is_complete()
    
    def full_goal_complete(self):
        for agent in self.agents:
            if not agent.is_complete():
                return False
        return True

    def generate_all_possible_functions(self) -> list[Function]:
        functions = []
        for function_schema in self.function_schemas:
            # get all possible values
            values = []
            if function_schema.type == ValueType.ENUMERATE:
                values = function_schema.range
            else:
                min, max = function_schema.range
                values = list(range(min, max + 1))
            
            # get all possible parameters
            all_entities = {}
            for key_word, type in function_schema.require_parameters.items():
                all_entities[key_word] = self.get_all_entity_name_by_type(type)
            
            keys = all_entities.keys()
            entities = all_entities.values()
            combinations = [comb for comb in product(*entities) if len(set(comb)) == len(comb)]
            all_entities = [dict(zip(keys, comb)) for comb in combinations]

            for value in values:
                for entity in all_entities:
                    new_function = Function()
                    new_function.name = function_schema.name
                    new_function.range = function_schema.range
                    new_function.type = function_schema.type
                    new_function.value = value
                    new_function.parameters = entity
                    functions.append(new_function)

        return functions


    def __str__(self):
        result = f"================= Model Result================\n"
        result += f"Domain name: {self.domain_name}\nProblem name: {self.problem_name}\n"
        # result += util.BIG_DIVIDER
        result += f"Problem type: {self.problem_type}\n"
        result += util.BIG_DIVIDER
        # result += f"Entities:\n"
        for entity in self.entities:
            result += f"{entity}\n"
        result += util.BIG_DIVIDER
        # result += f"Function Schemas:\n"
        for function_schema in self.function_schemas:
            result += f"{function_schema}\n"
        result += util.BIG_DIVIDER
        # result += f"Ontic Functions:\n"
        for ontic_function in self.ontic_functions:
            result += f"{ontic_function}\n"
        result += util.BIG_DIVIDER
        # result += f"Action Schemas:\n"
        for action_schema in self.action_schemas:
            result += util.SMALL_DIVIDER
            result += f"{action_schema}\n"
        result += util.BIG_DIVIDER
        # result += f"Agents:\n"
        for agent in self.agents:
            # result += util.SMALL_DIVIDER
            result += f"{agent}\n"
        return result

    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        new_model = Model()
        new_model.logger = self.logger
        new_model.observation_function = self.observation_function
        new_model.strategy = self.strategy
        new_model.problem_type = self.problem_type

        new_model.domain_name = self.domain_name
        new_model.problem_name = self.problem_name
        new_model.function_schemas = self.function_schemas
        new_model.action_schemas = self.action_schemas
        new_model.entities = self.entities
        new_model.ontic_functions = copy.deepcopy(self.ontic_functions)
        for agent in self.agents:
            new_model.agents.append(agent.copy())
        
        return new_model