from enum import Enum
import logging
import util
from itertools import combinations, product, permutations
import copy
from dataclasses import dataclass, field

MODEL_LOGGER_LEVEL = logging.DEBUG

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
        result = "Condition:\n"
        result += f"ep_operator: {self.ep_operator}, belief_sequence: {self.belief_sequence}, ep_truth: {self.ep_truth}\n"
        result += f"condition_operator: {self.condition_operator}, condition_function_schema: {self.condition_function_schema}\n"
        result += f"value: {self.value} / target_function_schema: {self.target_function_schema}"
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
        result = "EffectSchema:\n"
        result += f"effect_type: {self.effect_type}\n"
        result += f"effect_function_schema: {self.effect_function_schema}\n"
        result += f"value: {self.value}"
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
            result += util.SMALL_DIVIDER
            result += f"{pre_condition_schema}\n"
        # result += util.MEDIUM_DIVIDER
        result += f"Effect Schemas:\n"
        for effect_schema in self.effect_schemas:
            result += util.SMALL_DIVIDER
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
        result = "Goal:\n"
        result += f"ep_operator: {self.ep_operator}, belief_sequence: {self.belief_sequence}, ep_truth: {self.ep_truth}\n"
        result += f"condition_operator: {self.condition_operator}, goal_function_name: {self.goal_function_name}, goal_function_parameters: {self.goal_function_parameters}\n"
        result += f"value: {self.value} / target_function_name: {self.target_function_name}, target_function_parameters: {self.target_function_parameters}\n"
        return result

    def __repr__(self):
        return self.__str__()

class Agent:
    def __init__(self):
        self.name: str = None
        self.functions: list[Function] = []
        self.Goals: list[Goal] = []
    
    def is_valid_action(self, action: Action) -> bool:
        for condition in action.pre_condition:
            # TODO: 这里要加关于epistemic条件的判断
            checking_function = self.get_function_with_locator(condition.condition_function_locator)
            if checking_function is None:
                return False
            if condition.value is not None:
                if condition.value != checking_function.value:
                    return False
            else:
                target_function = self.get_function_with_locator(condition.target_function_locator)
                if target_function is None or checking_function.value != target_function.value:    
                    return False
            
        return True

    def get_function_with_locator(self, locator: FunctionLocator):
        for function in self.functions:
            if function.name == locator.name and list(function.parameters.values()) == list(locator.parameters.values()):
                return function
        return None
    
    def update_functions(self, functions: list[Function]):
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

    def __str__(self):
        result = f"Agent: {self.name}\n"
        result += util.MEDIUM_DIVIDER
        result += f"Functions:\n"
        for function in self.functions:
            result += f"{function}\n"
        result += util.MEDIUM_DIVIDER
        result += f"Goals:\n"
        for goal in self.Goals:
            result += util.SMALL_DIVIDER
            result += f"{goal}\n"
        return result

    def __repr__(self):
        return self.__str__()

class Model:
    def __init__(self, handler, observation_function_path):
        from absract_observation_function import AbsractObservationFunction
        self.logger = util.setup_logger(__name__, handler, logger_level=MODEL_LOGGER_LEVEL)
        ObsFunc = util.load_observation_function(observation_function_path, AbsractObservationFunction, self.logger)
        self.observation_function: AbsractObservationFunction = ObsFunc(handler)

        self.domain_name: str = None
        self.problem_name: str = None
        self.function_schemas: list[FunctionSchema] = []
        self.ontic_functions: list[Function] = []
        self.entities: list[Entity] = []
        self.action_schemas: list[ActionSchema] = []
        self.agents: list[Agent] = []
    
    def get_function_schema_by_name(self, name: str) -> FunctionSchema:
        return copy.deepcopy(next((function_schema for function_schema in self.function_schemas if function_schema.name == name), None))
    
    def get_all_entity_name_by_type(self, type: str) -> list[str]:
        """
        get all entity name by type
        """
        return [entity.name for entity in self.entities if entity.type == type]

    def run_a_round(self):
        for agent in self.agents:
            observe_functions = self.observation_function.get_observable_functions(model=self, agent=agent)
            # update agent's functions
            agent.update_functions(observe_functions)
            
            # get agent's successors
            agent_successors = self.get_agent_successors(agent)


    def get_agent_successors(self, agent: Agent) -> list[Action]:
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
            if agent.is_valid_action(action):
                result.append(action)
        return result

    def __str__(self):
        result = f"================= Model Result================\n"
        result += f"Domain name: {self.domain_name}\nProblem name: {self.problem_name}\n"
        result += util.BIG_DIVIDER
        result += f"Entities:\n"
        for entity in self.entities:
            result += f"{entity}\n"
        result += util.BIG_DIVIDER
        result += f"Function Schemas:\n"
        for function_schema in self.function_schemas:
            result += f"{function_schema}\n"
        result += util.BIG_DIVIDER
        result += f"Ontic Functions:\n"
        for ontic_function in self.ontic_functions:
            result += f"{ontic_function}\n"
        result += util.BIG_DIVIDER
        result += f"Action Schemas:\n"
        for action_schema in self.action_schemas:
            result += util.MEDIUM_DIVIDER
            result += f"{action_schema}\n"
        result += util.BIG_DIVIDER
        result += f"Agents:\n"
        for agent in self.agents:
            result += util.MEDIUM_DIVIDER
            result += f"{agent}\n"
        return result

    def __repr__(self):
        return self.__str__()