from enum import Enum
from collections import defaultdict
import util



class Model:
    def __init__(self):
        self.domain_name = None
        self.problem_name = None
        self.entities: list[Entity] = []
        self.base_states: list[BaseState] = []
        self.actions: list[BaseAction] = []
        self.ranges: list[Range] = []
        self.agents: list[Agent] = []
    
    def __str__(self):
        result = f"==================== Model Parsing Result ====================\n"
        result += f"Domain name: {self.domain_name}\n"
        result += f"Problem name: {self.problem_name}\n"
        result += util.BIG_DIVIDER
        result += f"Entities:\n"
        for entity in self.entities:
            result += f"{entity}\n"
        result += util.BIG_DIVIDER
        result += f"Base States:\n"
        for base_state in self.base_states:
            result += f"{base_state}\n"
        result += util.BIG_DIVIDER
        result += f"Actions:\n"
        for action in self.actions:
            result += util.SMALL_DIVIDER
            result += f"{action}\n"
        result += util.BIG_DIVIDER
        result += f"Ranges:\n"
        for range in self.ranges:
            result += f"{range}\n"
        result += util.BIG_DIVIDER
        result += f"Agents:\n"
        for agent in self.agents:
            result += util.SMALL_DIVIDER
            result += f"{agent}\n"
        
        return result
    
    def __repr__(self):
        return self.__str__()



class Agent:
    def __init__(self):
        self.name = None
        self.states: list[State] = []
        self.goals: list[ConditionUnit | EpistemicConditionUnit] = []
    
    def __str__(self):
        result = f"Agent: {self.name}\n"
        result += f"States:\n"
        for state in self.states:
            result += f"{state}\n"
        result += f"Goals:\n"
        for goal in self.goals:
            result += f"{goal}\n"
        return result

class Entity:
    def __init__(self):
        self.name = None
        self.type: str = None
    
    def __str__(self):
        return f"Entity({self.name} - {self.type})"

    def __repr__(self):
        return self.__str__()


class QuantityOperator(Enum):
    NONE = 0
    EQUAL = 1
    NOT_EQUAL = 2
    GREATER = 3
    GREATER_EQUAL = 4
    LESS = 5
    LESS_EQUAL = 6

class EpistemicTruthOperator(Enum):
    NONE = 0
    EQUAL = 1
    NOT_EQUAL = 2

class EpistemicTruth(Enum):
    NONE = 0
    TRUE = 1
    FALSE = 2
    UNKNOWN = 3

EPISTEMIC_TRUTH = {
    'true': EpistemicTruth.TRUE,
    'false': EpistemicTruth.FALSE,
    'unknown': EpistemicTruth.UNKNOWN
}

QUANTITY_OPERATOR = {
    '=': QuantityOperator.EQUAL,
    '!=': QuantityOperator.NOT_EQUAL,
    '>': QuantityOperator.GREATER,
    '>=': QuantityOperator.GREATER_EQUAL,
    '<': QuantityOperator.LESS,
    '<=': QuantityOperator.LESS_EQUAL
}

EPISTEMIC_TRUTH_OPERATOR = {
    '=': EpistemicTruthOperator.EQUAL,
    '!=': EpistemicTruthOperator.NOT_EQUAL
}

class LogicType(Enum):
    NONE = 0
    AND = 1
    OR = 2


class ConditionUnit:
    """
    A condition unit shows the structure of a single condition.\n
    Contains an operator: ConditionOperator\n
    state: BaseState\n
    value / target_variable: BaseState\n
    """
    def __init__(self):
        self.operator: QuantityOperator = QuantityOperator.NONE
        self.variable: BaseState = BaseState()
        self.value = None
        self.target_variable: BaseState = BaseState()

    def __str__(self):
        if self.value is not None:
            return f"ConditionUnit({self.variable} {self.operator} {self.value})"
        return f"ConditionUnit({self.variable} {self.operator} {self.target_variable})"

    def __repr__(self):
        return self.__str__()
    
class EpistemicConditionUnit(ConditionUnit):
    def __init__(self):
        super().__init__()
        self.epistemic_logic_operator: EpistemicTruthOperator = EpistemicTruthOperator.NONE
        self.belief_sequence: list[str] = []
        self.epistemic_truth = None
    
    def __str__(self):
        return f"EpistemicConditionUnit({self.epistemic_logic_operator} {self.belief_sequence} {super().__str__()} {self.epistemic_truth})"

    def __repr__(self):
        return self.__str__()


class ConditionBlock:
    """
    Contains a list of condition units and a logic type.\n
    The logic type can only be: AND or OR
    """
    def __init__(self):
        self.logic_type: LogicType = LogicType.NONE
        self.conditions: list[ConditionUnit] = []


class FullCondition:
    """
    A full condition will only exsist with the action.\n
    A condition represent the full pre-condition of an action.\n
    A condition build by multiple condition blocks.\n
    A condition block contains a list of condition units and a logic type. to illustrate the logic between conditions.\n
    A condition unit shows the structure of a single condition.
    """
    def __init__(self):
        self.blocks: list[ConditionBlock] = []


class BaseState:
    def __init__(self):
        self.name: str = None
        self.require_entities: dict[str, list[str]] = defaultdict(list)
        self.entities: list[str] = []
    
    def __str__(self):
        return f"BaseState(name: {self.name}, require_entities: {self.require_entities}, entities: {self.entities})"

    def __repr__(self):
        return self.__str__()


class State(BaseState):
    def __init__(self):
        super().__init__()
        self.value: int | float | str = None
        
    def __eq__(self, other):
        if isinstance(other, State):
            return self.name == other.name and self.entities == other.entities
        return False

    def __str__(self):
        return f"State({super().__str__()}, value: {self.value})"

    def __repr__(self):
        return super.__repr__()

class EffectOperator(Enum):
    NONE = 0
    ASSIGN = 1
    INCREASE = 2
    DECREASE = 3


EFFECT_OPERATOR = {
    'assign': EffectOperator.ASSIGN,
    'increase': EffectOperator.INCREASE,
    'decrease': EffectOperator.DECREASE
}

class EffectUnit:
    def __init__(self):
        self.operator: EffectOperator = EffectOperator.NONE
        self.value = None
        self.target_variable: BaseState = BaseState()
        self.variable: BaseState = BaseState()
    
    def __str__(self):
        if self.value is not None:
            return f"Effect('{self.operator}' {self.variable} {self.value})"
        return f"Effect(effect_operator:'{self.operator}' {self.variable} {self.target_variable})"

    def __repr__(self):
        return self.__str__()


class FullEffect:
    def __init__(self):
        self.effects: list[EffectUnit] = []


class BaseAction:
    def __init__(self):
        self.name = None
        self.parameters: dict[str, list[str]] = dict()
        self.pre_conditions: list[ConditionUnit | EpistemicConditionUnit] = []
        self.effects: list[EffectUnit] = []
    
    def __str__(self):
        result = f"Action:\n"
        result += f"Name: {self.name}\n"
        result += "Parameters:\n"
        for param_type, params in self.parameters.items():
            result += f"    {param_type} : {params}\n"
        result += "Preconditions:\n"
        count = 1
        for pre_condition in self.pre_conditions:
            result += f"{count}: {pre_condition}\n"
            count += 1
        count = 1
        result += "Effects:\n"
        for effect in self.effects:
            result += f"{count}: {effect}\n"
            count += 1
        return result

class RangeType(Enum):
    NONE = 0
    INTEGER = 1
    FLOAT = 2
    ENUMERATE = 3

RANGE_TPYE = {
    'integer': RangeType.INTEGER,
    'float': RangeType.FLOAT,
    'enumerate': RangeType.ENUMERATE
}

class Range:
    def __init__(self):
        self.name = None
        self.type: RangeType = RangeType.NONE
        self.enumerates: list[str] = []
        self.min = None
        self.max = None
    
    def __str__(self):
        return f"Range(name: {self.name}, type: {self.type}, min: {self.min}, max: {self.max}, enumerates: {self.enumerates})"