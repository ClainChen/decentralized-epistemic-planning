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

    def __str__(self):
        params = [f"{key} - {value}" for key, value in self.parameters.items()]
        return f"Function({self.name} {params} = {self.value})"

    def __repr__(self):
        return self.__str__()

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
        condition.belief_sequence = condition_schema.belief_sequence
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

    def __str__(self):
        if not self.value is None:
            result = f"Condition(({self.ep_operator} {self.belief_sequence} {self.ep_truth}) {self.condition_operator} {self.condition_function_name} {self.condition_function_parameters} = {self.value})"
        else:
            result = f"Condition(({self.ep_operator} {self.belief_sequence} {self.ep_truth}) {self.condition_operator} ({self.condition_function_name} {self.condition_function_parameters}) = ({self.target_function_name} {self.target_function_parameters}))"

        
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
        if not self.value is None:
            result = f"EffectSchema({self.effect_type} ({self.effect_function_schema.name} {self.effect_function_schema.require_parameters}) = {self.value})"
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
    
    @classmethod
    def init_with_schema_and_params(cls, effect_schema: EffectSchema, parameters: dict[str, Entity]):
        effect = cls()
        effect.effect_type = effect_schema.effect_type
        effect.effect_function_name = effect_schema.effect_function_schema.name
        effect.effect_function_parameters = {key: parameters[key] for key in effect_schema.effect_function_schema.require_parameters.keys()}
        if not effect_schema.value is None:
            effect.value = effect_schema.value
        else:
            effect.target_function_name = effect_schema.target_function_schema.name
            effect.target_function_parameters = {key: parameters[key] for key in effect_schema.target_function_schema.require_parameters.keys()}
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
    def __init__(self, action_schema: ActionSchema, parameters: dict[str, str]):
        self.name = action_schema.name
        self.parameters: dict[str, str] = parameters
        self.pre_condition: list[Condition] = []
        for condition_schema in action_schema.pre_condition_schemas:
            self.pre_condition.append(Condition.init_with_schema_and_params(condition_schema, parameters))
        self.effect: list[Effect] = []
        for effect_schema in action_schema.effect_schemas:
            self.effect.append(Effect.init_with_schema_and_params(effect_schema, parameters))

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

class Agent:
    def __init__(self):
        self.name: str = None
        self.functions: list[Function] = []
        self.goals: list[Condition] = []
        self.history_functions: list[list[Function]] = []

    def copy(self):
        new_agent = Agent()
        new_agent.name = self.name
        new_agent.goals = self.goals
        new_agent.functions = copy.deepcopy(self.functions)
        new_agent.history_functions = copy.deepcopy(self.history_functions)
        return new_agent

    def update_functions(self, functions: list[Function]):
        self.history_functions.append(copy.deepcopy(self.functions))
        if len(self.history_functions) >= 20:
            self.history_functions.pop(0)

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

    def is_complete(self, obs_func):
        for goal in self.goals:
            if not util.check_condition(goal, self.functions, self.history_functions, obs_func):
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
        return result

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

        self.domain_name: str = None
        self.problem_name: str = None
        self.function_schemas: list[FunctionSchema] = []
        self.ontic_functions: list[Function] = []
        self.entities: list[Entity] = []
        self.action_schemas: list[ActionSchema] = []
        self.agents: list[Agent] = []
    
    def init(self, handler, problem_type, observation_function_path, policy_strategy_path, rules_path):
        from abstracts import AbstractObservationFunction, AbstractPolicyStrategy, AbstractRules
        self.logger = util.setup_logger(__name__, handler, logger_level=MODEL_LOGGER_LEVEL)
        
        ObsFunc = util.load_observation_function(observation_function_path, self.logger)
        self.observation_function: AbstractObservationFunction = ObsFunc(handler)
        self.logger.info(f"Loaded observation function: {ObsFunc.__name__}")

        Strategy = util.load_policy_strategy(policy_strategy_path, self.logger)
        self.strategy: AbstractPolicyStrategy = Strategy(handler)
        self.logger.info(f"Loaded policy strategy: {Strategy.__name__}")

        Rules = util.load_rules(rules_path, self.logger)
        self.rules: AbstractRules = Rules(handler)
        self.logger.info(f"Loaded rules: {Rules.__name__}")

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

    def get_all_agent_names(self) -> list[str]:
        return [agent.name for agent in self.agents]

    def simulate(self):
        """
        Simulate the model until all agents have reached a terminal state
        """
        agent_index = 0
        agent_count = len(self.agents)
        # let all agent observe once to get the initial observation
        for agent in self.agents:
            self.observe_and_update_agent(agent.name)
        while not self.full_goal_complete():
            self.agent_decide_action_and_move(self.agents[agent_index].name)
            agent_index = (agent_index + 1) % agent_count

    def agent_decide_action_and_move(self, agent_name: str):
        self.logger.debug(f"{agent_name} moving:\n{self}")
        self.observe_and_update_agent(agent_name)
        action = self.strategy.get_policy(self, agent_name)
        self.agent_move(agent_name, action)
        # TODO: #5 需要思考一下如何进行intention prediction
    
    def agent_move(self, agent_name: str, action: Action):
        if action is not None and util.is_valid_action(self.ontic_functions, action):
            if not self.do_action(agent_name, action):
                print(f"{agent_name} cannot take action: {action.name}, takes action: stay")
            else:
                print(f"{agent_name} takes action: {action.name}")
        else:
            self.do_action(agent_name, None)
            print(f"{agent_name} takes action: stay")

    def observe_and_update_agent(self, agent_name: str):
        agent = self.get_agent_by_name(agent_name)
        observe_functions = self.observation_function.get_observable_functions(self.ontic_functions, agent_name)
        # update agent's functions
        agent.update_functions(observe_functions)

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
        function = util.get_function_with_name_and_params(functions, effect.effect_function_name, effect.effect_function_parameters)
        if function is not None:
            if not effect.value is None:
                function.value = util.update_effect_value(function.value, effect.value, effect.effect_type)
            else:
                target_function = util.get_function_with_name_and_params(self.ontic_functions, effect.target_function_name, effect.target_function_parameters)
                function.value = util.update_effect_value(function.value, target_function.value, effect.effect_type)
        else:
            if effect.effect_type != EffectType.ASSIGN:
                self.logger.error(f"Trying to change a function that not exist in agent's functions or ontic world functions")
                raise ValueError("Trying to change a function that not exist in agent's functions or ontic world functions")
            function = Function()
            func_schema = self.get_function_schema_by_name(effect.effect_function_name)
            function.name = effect.effect_function_name
            function.parameters = effect.effect_function_parameters
            function.range = func_schema.range
            function.type = func_schema.type

            if not effect.value is None:
                function.value = effect.value
                functions.append(function)
            else:
                target_function = util.get_function_with_name_and_params(self.ontic_functions, effect.effect_function_name, effect.effect_function_parameters)
                if target_function is None:
                    self.logger.error(f"Target function {effect.target_function_locator.name} not found in effect phase")
                    raise ValueError(f"Target function {effect.target_function_locator.name} not found in effect phase")

                function.value = target_function.value
                functions.append(function)
        if not util.check_in_range(function):
            print(functions)
            print(effect)
            raise ValueError(f"{function.name} is out of range")

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
            if util.is_valid_action(agent.functions, action, agent.history_functions, self.observation_function, is_ontic_checking=False):
                result.append(action)
        return result

    def agent_goal_complete(self, agent_name: str):
        agent = self.get_agent_by_name(agent_name)
        return agent.is_complete(self.observation_function)
    
    def full_goal_complete(self):
        if not self.agents:
            raise ValueError("No agents in the model")
    
        if self.observation_function is None:
            raise ValueError("Observation function is not initialized")
    
        if self.problem_type == ProblemType.COOPERATIVE:
            return any(agent.is_complete(self.observation_function) for agent in self.agents)
        else:
            return all(agent.is_complete(self.observation_function) for agent in self.agents)

    def get_belief_functions_of_agent_with_belief_sequence(self, agent_name: str, belief_sequence: list[str]) -> list[Function]:
        ontic_functions = self.ontic_functions
        belief_sequence = [agent_name] + belief_sequence
        current_agent = agent_name
        for agent in belief_sequence:
            observe_result = self.observation_function.get_observable_functions(ontic_functions, current_agent)
            ontic_functions = observe_result[current_agent]
            current_agent = agent

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
        new_model.rules = self.rules
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