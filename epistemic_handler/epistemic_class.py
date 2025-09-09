from enum import Enum
import logging
import util
from itertools import combinations, product, permutations, chain
import copy
from dataclasses import dataclass, field
from collections import defaultdict

MODEL_LOGGER_LEVEL = logging.DEBUG
class ProblemType(Enum):
    COOPERATIVE = 1
    NEUTRAL = 2

    def __str__(self):
        return f"\"{self.name}\""
    
    def __repr__(self):
        return self.__str__()

class EpistemicOperator(Enum):
    EQUAL = 1
    NOT_EQUAL = 2
    NONE = 0

    def __str__(self):
        return f"\"EP_{self.name}\""

    def __repr__(self):
        return self.__str__()

class EpistemicTruth(Enum):
    TRUE = 1
    FALSE = 2
    UNKNOWN = 3
    NONE = 0

    def __str__(self):
        return f"\"EP_{self.name}\""

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
        mapper = {
            ValueType.INTEGER: "integer",
            ValueType.ENUMERATE: "enumerate",
            ValueType.NONE: "none"
        }
        return mapper[self]

    def __repr__(self):
        return self.__str__()

class GoalValueType(Enum):
    ENUMERATE = 1
    RANGE = 2
    ALL = 3
    NONE = 4

    def __str__(self):
        mapper = {
            GoalValueType.ENUMERATE: "enumerate",
            GoalValueType.RANGE: "range",
            GoalValueType.ALL: "all",
            GoalValueType.NONE: "none"
        }
        return mapper[self]

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
    "unknown": EpistemicTruth.UNKNOWN,
    "none": EpistemicTruth.NONE
}

CONDITION_OPERATOR_MAPS = {
    "=": ConditionOperator.EQUAL,
    "!=": ConditionOperator.NOT_EQUAL,
    ">": ConditionOperator.GREATER,
    "<": ConditionOperator.LESS,
    ">=": ConditionOperator.GREATER_EQUAL,
    "<=": ConditionOperator.LESS_EQUAL,
    "": ConditionOperator.NONE
}

EFFECT_TYPE_MAPS = {
    "assign": EffectType.ASSIGN,
    "increase": EffectType.INCREASE,
    "decrease": EffectType.DECREASE,
}

GOAL_VALUE_TYPE_MAPS = {
    'enumerate': GoalValueType.ENUMERATE,
    'range': GoalValueType.RANGE,
    'all': GoalValueType.ALL,
    'none': GoalValueType.NONE
}

@dataclass
class Entity:
    name: str = None
    type: str = None

    def __str__(self):
        return f"Entity({self.name} - {self.type})"
    
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

    def header(self):
        return f"{self.name}({list(self.parameters.values())})"

    def __str__(self):
        return f"Function({self.name} {list(self.parameters.values())} = {self.value})"

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, Function):
            return False
        return (self.name == other.name and 
                self.parameters == other.parameters and 
                self._value == other._value)
     
    def __hash__(self):
        return hash((self.name, frozenset(self.parameters), self._value))
    
    def plain_text(self):
        return f"{self.header()}={self.value}"
    
    def __lt__(self, other):
        return self.plain_text() < other.plain_text()

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
        if not self.value is None:
            result = f"ConditionSchema(({self.ep_operator} {self.belief_sequence} {self.ep_truth}) {self.condition_operator} ({self.condition_function_schema.name} {self.condition_function_schema.require_parameters}) = {self.value})"
        else:
            result = f"ConditionSchema(({self.ep_operator} {self.belief_sequence} {self.ep_truth}) {self.condition_operator} ({self.condition_function_schema.name} {self.condition_function_schema.require_parameters}) = ({self.target_function_schema.name} {self.target_function_schema.require_parameters}))"
        
        return result

    def __repr__(self):
        return self.__str__()

class Condition:
    def __init__(self):
        self.ep_operator: EpistemicOperator = EpistemicOperator.NONE
        self.belief_sequence: list[str] = []
        self.ep_truth: EpistemicTruth = EpistemicTruth.NONE
        self.condition_operator: ConditionOperator = ConditionOperator.NONE
        self.condition_function_name: str = None
        self.condition_function_parameters: dict[str, str] = None
        self.target_function_name: str = None
        self.target_function_parameters: dict[str, str] = None
        self.value: int | str = None

    @classmethod
    def init_with_schema_and_params(cls, condition_schema: ConditionSchema, parameters: dict[str, str]):
        condition = cls()
        condition.ep_operator = condition_schema.ep_operator
        condition.belief_sequence = [parameters[key] for key in condition_schema.belief_sequence]
        condition.ep_truth = condition_schema.ep_truth
        condition.condition_operator = condition_schema.condition_operator
        condition.condition_function_name = condition_schema.condition_function_schema.name
        condition.condition_function_parameters = {key: parameters[key] for key in condition_schema.condition_function_schema.require_parameters.keys()}
        if not condition_schema.value is None:
            condition.value = condition_schema.value
        else:
            condition.target_function_name = condition_schema.target_function_schema.name
            condition.target_function_parameters = {key: parameters[key] for key in condition_schema.target_function_schema.require_parameters.keys()}
        return condition
    
    @classmethod
    def create_with_function_and_belief_sequence(cls, function: Function, belief_sequence: list[str]):
        condition = cls()
        if len(belief_sequence) > 0:
            condition.ep_operator = EpistemicOperator.EQUAL
            condition.ep_truth = EpistemicTruth.TRUE
            condition.belief_sequence = belief_sequence
        condition.condition_operator = ConditionOperator.EQUAL
        condition.condition_function_name = function.name
        condition.condition_function_parameters = function.parameters
        condition.value = function.value
        return condition

    def __str__(self):
        if not self.value is None:
            result = f"Condition({list(self.belief_sequence)} {self.condition_function_name} {list(self.condition_function_parameters.values())} = {self.value})"
        else:
            result = f"Condition({list(self.belief_sequence)} {self.condition_function_name} {list(self.condition_function_parameters.values())} = {self.target_function_name} {list(self.target_function_parameters.values())})"

        
        return result

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Condition):
            return False
        return (self.ep_operator == other.ep_operator and
                tuple(self.belief_sequence) == tuple(other.belief_sequence) and
                self.ep_truth == other.ep_truth and
                self.condition_operator == other.condition_operator and
                self.condition_function_name == other.condition_function_name and
                frozenset(self.condition_function_parameters.items() if self.condition_function_parameters else []) == frozenset(other.condition_function_parameters.items() if other.condition_function_parameters else []) and
                self.target_function_name == other.target_function_name and
                frozenset(self.target_function_parameters.items() if self.target_function_parameters else []) == frozenset(other.target_function_parameters.items() if other.target_function_parameters else []) and
                self.value == other.value)

    def __hash__(self):
        return hash((
            self.ep_operator,
            tuple(self.belief_sequence),
            self.ep_truth,
            self.condition_operator,
            self.condition_function_name,
            frozenset(self.condition_function_parameters.items() if self.condition_function_parameters else []),
            self.target_function_name,
            frozenset(self.target_function_parameters.items() if self.target_function_parameters else []),
            self.value
        ))
    
    def header(self):
        return f"{self.condition_function_name}({list(self.condition_function_parameters.values())})"

    def header_with_bs(self):
        return f"{self.condition_function_name}({self.belief_sequence})({list(self.condition_function_parameters.values())})"
    
    def plain_text(self):
        return f"{self.header()}={self.value}"

class EffectSchema:
    def __init__(self):
        self.effect_type: EffectType = EffectType.NONE
        self.effect_function_schema: FunctionSchema = None
        self.value: int | str = None
        self.target_function_schema: FunctionSchema = None
        self.target_function_belief_sequence: list[str] = []

    def __str__(self):
        if not self.value is None:
            result = f"EffectSchema({self.effect_type} ({self.effect_function_schema.name} {self.effect_function_schema.require_parameters}) = {self.value})"
        elif len(self.target_function_belief_sequence) > 0:
            result = f"EffectSchema({self.effect_type} ({self.effect_function_schema.name} {self.effect_function_schema.require_parameters}) = ({self.target_function_belief_sequence} {self.target_function_schema.name} {self.target_function_schema.require_parameters}))"
        else:
            result = f"EffectSchema({self.effect_type} ({self.effect_function_schema.name} {self.effect_function_schema.require_parameters}) = ({self.target_function_schema.name} {self.target_function_schema.require_parameters}))"
        return result
    
    def __repr__(self):
        return self.__str__()

class Effect:
    def __init__(self):
        self.effect_type: EffectType = EffectType.NONE
        self.effect_function_name: str = None
        self.effect_function_parameters: dict[str, Entity] = {}
        self.target_function_name: str = None
        self.target_function_parameters: dict[str, Entity] = {}
        self.value: int | str = None
        self.target_function_belief_sequence: list[str] = []
    
    @classmethod
    def init_with_schema_and_params(cls, effect_schema: EffectSchema, parameters: dict[str, Entity]):
        effect = cls()
        effect.effect_type = effect_schema.effect_type
        effect.effect_function_name = effect_schema.effect_function_schema.name
        effect.effect_function_parameters = {key: parameters[key] for key in effect_schema.effect_function_schema.require_parameters.keys()}
        if effect_schema.value is not None:
            effect.value = effect_schema.value
        else:
            effect.target_function_name = effect_schema.target_function_schema.name
            effect.target_function_parameters = {key: parameters[key] for key in effect_schema.target_function_schema.require_parameters.keys()}
            effect.target_function_belief_sequence = [parameters[key] for key in effect_schema.target_function_belief_sequence]
        return effect

    def __str__(self):
        if not self.value is None:
            result = f"Effect({self.effect_type} ({self.effect_function_name} {self.effect_function_parameters}) = {self.value})"
        else:
            result = f"Effect({self.effect_type} ({self.effect_function_name} {self.effect_function_parameters}) = ({self.target_function_name} {self.target_function_parameters}))"
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
    def __init__(self):
        self.name = ""
        self.parameters: dict[str, str] = {}
        self.pre_condition: list[Condition] = []
        self.effect: list[Effect] = []

    @classmethod
    def stay_action(cls, agent_name: str):
        result = cls()
        result.name = "stay"
        result.parameters['?self'] = agent_name
        return result
    
    @classmethod
    def create_action(cls, action_schema: ActionSchema, parameters: dict[str, str]) -> 'Action':
        result = cls()
        result.name = action_schema.name
        result.parameters = parameters
        result.pre_condition = []
        for condition_schema in action_schema.pre_condition_schemas:
            result.pre_condition.append(Condition.init_with_schema_and_params(condition_schema, parameters))
        result.effect = []
        for effect_schema in action_schema.effect_schemas:
            result.effect.append(Effect.init_with_schema_and_params(effect_schema, parameters))
        return result

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

    def __hash__(self):
        return hash(
            (
                self.name,
                frozenset(self.parameters)
            )
        )

    def __eq__(self, value):
        if isinstance(value, Action):
            return self.name == value.name and self.parameters == value.parameters
        else:
            return False

    def __repr__(self):
        return self.__str__()
    
    def header(self) -> str:
        return f"{self.name}({list(self.parameters.values())})"

    def __lt__(self, other):
        return self.header() < other.header()

class Agent:
    def __init__(self):
        self.name: str = None
        self.own_goals: list[Condition] = []
        self.other_goals: dict[str, list[Condition]] = {} # This only use in cooperative problem

        # The following are only useful in neutral problem
        self.complete_signal = False
        self.all_possible_goals: list[dict[str, list[Condition]]] = []
        self.max_time = float('inf')

        # E: world x Agt -> Act
        # the experiences (belief of actions in world) of this action
        self.E: dict[frozenset, dict[str, set[Action]]] = defaultdict(lambda: defaultdict(set))

    def copy(self):
        new_agent = Agent()
        new_agent.name = self.name
        new_agent.own_goals = copy.deepcopy(self.own_goals)
        new_agent.other_goals = copy.deepcopy(self.other_goals)
        new_agent.complete_signal = self.complete_signal
        new_agent.all_possible_goals = copy.deepcopy(self.all_possible_goals)
        new_agent.E = copy.deepcopy(self.E)
        return new_agent

    def __str__(self):
        result = f"Agent: {self.name}\nMax move time: {self.max_time}\n"
        result += f"Goal completed: \'{self.complete_signal}\'\n"
        result += f"Own Goals:\n"
        for goal in self.own_goals:
            result += f"{goal}\n"
        result += f"Other Goals:\n"
        for agent, goals in self.other_goals.items():
            result += f"{agent}:\n"
            for goal in goals:
                result += f"{goal}\n"
        result += f"All Possible Goals:\n"
        for goal_set in self.all_possible_goals:
            for name, goals in goal_set.items():
                result += f"{name}:\n"
                for goal in goals:
                    result += f"{goal}\n"
            result += f"{util.SMALL_DIVIDER}"
        return result

    def __repr__(self):
        return self.__str__()

class AcceptableGoal:
    from epistemic_handler.file_parser import ParsingAcceptableGoal
    def __init__(self):
        self.function_name = ""
        self.type = GoalValueType.NONE
        self.parameters: list[str] = []
        self.enumerates: list = []
        self.min = None
        self.max = None
    
    def __str__(self):
        return f"AcceptableGoal(function_name: {self.function_name}, type: {self.type}, enumerates: {self.enumerates}, min: {self.min}, max: {self.max})"

    def __repr__(self):
        return self.__str__()

class Model:
    def __init__(self):
        from abstracts import AbstractObservationFunction, AbstractPolicyStrategy, AbstractRules
        self.logger = None
        self.observation_function: AbstractObservationFunction = None
        self.strategy: AbstractPolicyStrategy = None
        self.rules: AbstractRules = None
        self.problem_type: ProblemType = None
        self.S_G: set[Condition] = set()
        self.max_belief_depth: int = 1
        self.possible_belief_sequences: list[list[str]] = []

        self.domain_name: str = None
        self.problem_name: str = None
        self.function_schemas: list[FunctionSchema] = []
        self.ontic_functions: list[Function] = []
        self.history: list[dict] = []
        self.entities: list[Entity] = []
        self.action_schemas: list[ActionSchema] = []
        self.agents: list[Agent] = []
    
    def init(self, handler, problem_type, observation_function_path, policy_strategy_path, rules_path):
        from abstracts import AbstractObservationFunction, AbstractPolicyStrategy, AbstractRules
        self.logger = util.setup_logger(__name__, handler, logger_level=MODEL_LOGGER_LEVEL)
        
        ObsFunc = util.load_observation_function(observation_function_path, self.logger)
        self.observation_function: AbstractObservationFunction = ObsFunc(handler)
        self.logger.info(f"Loaded observation function: {ObsFunc.__name__}")

        Rules = util.load_rules(rules_path, self.logger)
        self.rules: AbstractRules = Rules(handler)
        self.logger.info(f"Loaded rules: {Rules.__name__}")

        Strategy = util.load_policy_strategy(policy_strategy_path, self.logger)
        self.strategy: AbstractPolicyStrategy = Strategy(handler)
        self.logger.info(f"Loaded policy strategy: {Strategy.__name__}")

        self.problem_type: ProblemType = PROBLEM_TYPE_MAPS[problem_type]
        self.logger.info(f"Loaded problem type: {self.problem_type}")  
    
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

    def get_all_agent_names(self) -> list[str]:
        return [agent.name for agent in self.agents]

    def get_agent_successors(self, agent_name: str) -> list[Action]:
        result = []
        # if self.agent_goal_complete(agent_name):
        #     return result

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
                successor = Action.create_action(action_schema, param)
                candidates.append(successor)
        candidates.append(Action.stay_action(agent_name))
        for action in candidates:
            if util.is_valid_action(self, action):
                result.append(action)
        return sorted(result)

    def agent_goal_complete(self, agent_name: str):
        agent = self.get_agent_by_name(agent_name)
        goals = agent.own_goals.copy()
        for goal in goals:
            if not util.check_condition(self, goal):
                agent.complete_signal = False
                return False
        agent.complete_signal = True
        return True
    
    def full_goal_complete(self):
        if not self.agents:
            raise ValueError("No agents in the model")
    
        if self.observation_function is None:
            raise ValueError("Observation function is not initialized")
    
        # if self.problem_type == ProblemType.COOPERATIVE:
        #     return any(agent.is_complete(self.observation_function) for agent in self.agents)
        # else:
        #     return all(agent.is_complete(self.observation_function) for agent in self.agents)
        for agent in self.agents:
            self.agent_goal_complete(agent.name)
        return all(agent.complete_signal for agent in self.agents)
    
    def get_history_functions(self) -> list[list[Function]]:
        """
        get all history functions
        """
        return [history['functions'] for history in self.history] + [self.ontic_functions]

    def get_functions_of_agent(self, agent_name: str) -> list[Function]:
        return self.observation_function.get_observable_functions(self, self.ontic_functions, agent_name)

    def get_history_functions_of_agent(self, agent_name: str) -> list[list[Function]]:
        result = []
        for history in self.history:
            result.append(self.observation_function.get_observable_functions(self, history['functions'], agent_name))
        return result

    def get_next_agent(self, current_agent: str) -> str:
        cur_index = self.get_agent_index_by_name(current_agent)
        return self.agents[(cur_index + 1) % len(self.agents)].name

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

    def update_belief_goals(self):
        # 只会在Neutral Mode中被调用
        # 如果agent能够看见其他agent，则会根据自己的观察来更新自己对其他agent的goals的信念
        # 每一个历史时间戳上都会有一个对应的history world和agent: action对，用于反映在某个世界下某个agent执行了某个action
        # 如果在那个时间戳时agent无法看见那个正在行动的agent，则他无法得知那个agent在那个时间戳的action
        # 如果观察到某个agent的complete signal为true，则会更好操作
        for a in self.agents:
            for b in self.agents:
                if a.name == b.name:
                    continue
                current_fb_fa = util.get_epistemic_world(self, [a.name,b.name])
                # only remain the acceptable functions
                acceptable_fb_fa = []
                for func in current_fb_fa:
                    for sg in self.S_G:
                        if (sg.condition_function_name == func.name and 
                            sg.condition_function_parameters == func.parameters and
                            sg.value == func.value):
                            acceptable_fb_fa.append(func)
                # generate all possible goal functions for the target agent
                possible_func_sets = list(chain.from_iterable(combinations(acceptable_fb_fa, r) for r in range(1, len(acceptable_fb_fa) + 1)))
                possible_func_sets = [set(func.plain_text() for func in goal_sets) for goal_sets in possible_func_sets]

                remain_possible_goals = []
                if not b.complete_signal:
                    for goal_set in a.all_possible_goals:
                        if set(cond.plain_text() for cond in goal_set[b.name]) not in possible_func_sets:
                            remain_possible_goals.append(goal_set)
                else:
                    for goal_set in a.all_possible_goals:
                        if set(cond.plain_text() for cond in goal_set[b.name]) in possible_func_sets:
                            remain_possible_goals.append(goal_set)
                # print(f"len possi func sets: {len(possible_func_sets)}, len remain possi goals: {len(remain_possible_goals)}")
                if len(remain_possible_goals) > 0 :
                    a.all_possible_goals = remain_possible_goals

    def update_agent_belief_actions_in_world(self, last_agent, action):
        for agent in self.agents:
            if last_agent == agent.name:
                continue
            if last_agent not in self.observation_function.get_observable_agents(self, self.ontic_functions, agent.name):
                continue
            agent_last_jp_world = util.get_epistemic_world(self, [agent.name])
            hash_set_agent_last_jp_world = frozenset(agent_last_jp_world)
            agent.E[hash_set_agent_last_jp_world][last_agent].add(action)
            # if an agent found his done action is the same as the available actions in this environment
            # he will reset the belief actions of himself to empty
            # if last_agent == agent.name:
            #     this_succs = set(self.get_agent_successors(last_agent))
            #     if this_succs.issubset(agent.action_under_jp_worlds[hash_set_agent_last_jp_world][last_agent]):
            #         agent.action_under_jp_worlds[hash_set_agent_last_jp_world][last_agent].difference_update(this_succs)

    @util.record_time
    def simulate(self, start_agent = ""):
        """
        Simulate the model until all agents have reached a terminal state
        """
        if start_agent == "":
            agent_name = self.agents[0].name
        else:
            agent_name = start_agent
        while not self.full_goal_complete():
            # update the belief goals of each agent, and update their observed world
            if self.problem_type == ProblemType.NEUTRAL:
                self.update_belief_goals()

            # decide the action and do the action
            action = self.strategy.get_policy(self, agent_name)

            if self.problem_type == ProblemType.NEUTRAL:
                self.update_agent_belief_actions_in_world(agent_name, action)

            self.move(agent_name, action)

            # log
            output = f"{agent_name} takes action: {action.header()}"
            print(output)
            self.logger.info(output)
            # for agent in self.agents:
            #     self.logger.debug(agent.action_under_jp_worlds)

            agent_name = self.get_next_agent(agent_name)

        self.logger.info(f"{self.show_solution()}")
    
    def move(self, agent_name: str, action: Action):
        history = {'functions': copy.deepcopy(self.ontic_functions),
                   'agent': agent_name,
                   'action': action,
                   'signal': {agent.name: agent.complete_signal for agent in self.agents}}
        self.history.append(history)
        for effect in action.effect:
            self.update_functions(effect)

    def sim_move(self, agent_name, action):
        if self.full_goal_complete():
            print("The goal is complete")
            exit(0)

        # update the belief goals of each agent, and update their observed world
        if not util.is_valid_action(self, action):
            raise Exception(f"Invalid action: {action.header()}")

        if self.problem_type == ProblemType.NEUTRAL:
            self.update_belief_goals()
            self.update_agent_belief_actions_in_world(agent_name, action)

        self.move(agent_name, action)

        # log
        output = f"{agent_name} takes action: {action.header()}"
        print(output)
        self.logger.info(output)
              
    def update_functions(self, effect: Effect):
        function = util.get_function_with_name_and_params(self.ontic_functions, effect.effect_function_name, effect.effect_function_parameters)
        assert function is not None, f"updating function is not found!"
        if effect.value is not None:
            function.value = util.update_effect_value(function.value, effect.value, effect.effect_type)
        else:
            checking_functions = util.get_epistemic_world(self, effect.target_function_belief_sequence, [history['functions'] for history in self.history])
            target_function = util.get_function_with_name_and_params(checking_functions, effect.target_function_name, effect.target_function_parameters)
            function.value = util.update_effect_value(function.value, target_function.value, effect.effect_type)
        if not util.check_in_range(function):
            # print(functions)
            # print(effect)
            raise ValueError(f"{function.name} is out of range")

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
        # result += f"Action Schemas:\n"
        for action_schema in self.action_schemas:
            result += util.SMALL_DIVIDER
            result += f"{action_schema}\n"
        result += util.BIG_DIVIDER
        # result += f"Ontic Functions:\n"
        for ontic_function in self.ontic_functions:
            result += f"{ontic_function}\n"
        result += util.BIG_DIVIDER
        # result += f"Agents:\n"
        for agent in self.agents:
            # result += util.SMALL_DIVIDER
            result += f"{agent}\n"
        return result

    def __repr__(self):
        return self.__str__()
    
    def show_solution(self):
        result = "====== Solution ======\n"
        for i in range(len(self.history)):
            result += f"Step {i+1}:\n"
            history = self.history[i]
            result += f"{history['agent']}: {history['action'].header()}\n"
            result += f"{history['signal']}\n"            
            result += f"{util.SMALL_DIVIDER}"
        return result
    
    def copy(self):
        new_model = Model()
        new_model.logger = self.logger
        new_model.observation_function = self.observation_function
        new_model.strategy = self.strategy
        new_model.rules = self.rules
        new_model.problem_type = self.problem_type
        new_model.domain_name = self.domain_name
        new_model.problem_name = self.problem_name
        new_model.S_G = self.S_G
        new_model.max_belief_depth = self.max_belief_depth
        new_model.possible_belief_sequences = self.possible_belief_sequences
        new_model.function_schemas = self.function_schemas
        new_model.action_schemas = self.action_schemas
        new_model.entities = self.entities
        new_model.ontic_functions = copy.deepcopy(self.ontic_functions)
        new_model.history = copy.deepcopy(self.history)
        for agent in self.agents:
            new_model.agents.append(agent.copy())
        
        return new_model