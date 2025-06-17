from enum import Enum


class Model:
    def __init__(self):
        self.domain_name = None
        self.problem_name = None
        self.entities: list[Entity] = []
        self.base_states: list[BaseState] = []
        self.actions: list[BaseAction] = []


class EntityType(Enum):
    NONE = 0
    AGENT = 1
    OBJECT = 2


class Entity:
    def __init__(self):
        self.name = None
        self.type: EntityType = EntityType.NONE


class ConditionOperator(Enum):
    NONE = 0
    EQUAL = 1
    NOT_EQUAL = 2
    GREATER = 3
    GREATER_EQUAL = 4
    LESS = 5
    LESS_EQUAL = 6


class LogicType(Enum):
    NONE = 0
    AND = 1
    OR = 2


class ConditionUnit:
    def __init__(self):
        self.operator: ConditionOperator = ConditionOperator.NONE
        self.state: BaseState = BaseState()
        self.value = 0
        self.target_variable: State = State()


class ConditionBlock:
    def __init__(self):
        self.logic_type: LogicType = LogicType.NONE
        self.conditions: list[ConditionUnit] = []


class BaseState:
    def __init__(self):
        self.name = None
        self.entities = None
        self.range = []

    def __eq__(self, other):
        if isinstance(other, BaseState):
            return self.name == other.name and self.entities == other.entities
        return False


class State(BaseState):
    def __init__(self):
        super().__init__()
        self.value = 0


class EffectType(Enum):
    NONE = 0
    ASSIGN = 1
    INCREASE = 2
    DECREASE = 3


class EffectUnit:
    def __init__(self):
        self.type: EffectType = EffectType.NONE
        self.value = 0
        self.state: State = State()


class BaseAction:
    def __init__(self):
        self.name = None
        self.parameters: list[Entity] = []
        self.pre_conditions: list[ConditionUnit] = []
        self.effects: list[EffectUnit] = []
