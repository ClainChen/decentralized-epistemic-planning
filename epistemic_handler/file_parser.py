import logging
import copy
import os

import util

DOMAIN_LOG_LEVEL = logging.INFO
PROBLEM_LOG_LEVEL = logging.INFO
MODEL_CHECKER_LOG_LEVEL = logging.INFO

# Domain related regexes
DOMAIN_NAME_REGEX = r"\(domain (\w+)\)"
TYPES_REGEX = r"\(:types$\n([\s\S]+?)\n^\s+\)"
FUNCTION_EXTRACT_REGEX = r"\(:functions$\n([\s\S]+?)^\s+\)"
FUNTION_SPLIT_NAME_PARAMETER_REGEX = r"\((\w+) ([^\)]+)\)"
SPLIT_MULTI_PARAMETER_REGEX = r"([^-\(]+)- (\w+)"
ACTION_EXTRACT_REGEX = r"\(:action (\w+)\n\s+:parameters (.*)\n\s+:precondition \(\n(\s+\([\s\S]*?)?\s+\)\n\s+:effect \(\n(\s+\([\s\S]*?)?\s+\)\n\s+\)"
CONDITION_SPLIT_REGEX = r"\(([!=><]=?|>|<) \(([^\)]+)\) \(?(\d|[^\)]+)\){1,2}"
EPISTEMIC_CONDITION_SPLIT_REGEX = r"\(([!=><]=?|>|<) \(@ep \(\"([\s\S]+?)\"\) \(([!=><]=?|>|<) \(([^\)]+)\) \(?(\d|[^\)]+)\){1,2} ep.(\w+)\)"
SPLIT_VARIABLE_REGEX = r"(\w+) (.+)"
EXTRACT_BELIEF_AGENT_REGEX = r"\[(\w+)\]"
EFFECT_SPLIT_REGEX = r"\((increase|decrease|assign) \(([^\)]+)\) \(?(\d|[^\)]+)\){1,2}"

# Problem related regexes
PROBLEM_NAME_REGEX = r"\(problem (\w+)\)"
PROBLEM_DOMAIN_NAME_REGEX = r"\(:domain (\w+)\)"
ENV_AGENTS_REGEX = r"\(:agents\s*\n\s+(.+)\n\s+\)"
ENV_OBJECT_EXTRACT_REGEX = r"\(:objects$\n([\s\S]*?)^\s+\)"
ENV_OBJECT_SPLIT_REGEX = r"([^-]+)- (\w+)"
GOAL_REGEX = r"\(:goal \(and$\n([\s\S]+?)^\s+\)\n\s+\)"
RANGES_EXTRACT_REGEX = r"\(:ranges$\n([\s\S]+?)^\s+\)"
RANGES_SPLIT_REGEX = r"\((.+) (\w+) \[(.+)\]\)"
AGENT_INIT_REGEX = r"\(:init$\n([\s\S]*?)^\s+\)"
INIT_STATE_EXTRACT_REGEX = r"\(:init$\n([\s\S]*?)^\s+\)"
GOAL_SET_REGEX = r"\(:goal_sets\s*\n\s+(.+)\n\s+\)"
MAX_BELIEF_DEPTH_REGEX = r"\(:max_belief_depth (\d+)\)"
# SHARED_INIT_STATE_EXTRACT_REGEX = r"\(:shared-init$\n([\s\S]*?)^\s+\)"
INIT_STATE_SPLIT_REGEX = r"assign \((.+?)\) \(?('\w+'|\d*|.+?)\){1,2}"
AGENT_NAME_REGEX = r"\(:agent (\w+)\)"




class ParsingAction:
    """
    Action class for a better management during parsing
    """
    def __init__(self):
        self.name = None
        self.parameters: dict[str, list[str]] = dict()
        self.pre_conditions: list[ParsingCondition] = []
        self.effects: list[ParsingEffect] = []

    def __str__(self):
        result = f"Action:\n"
        result += f"Name: {self.name}\n"
        result += "Parameters:\n"
        for param_type, params in self.parameters.items():
            result += f"    {param_type} : {params}\n"
        result += "Preconditions:\n"
        count = 1
        for precondition in self.pre_conditions:
            result += f"{count}: {precondition}\n"
            count += 1
        count = 1
        result += "Effects:\n"
        for effect in self.effects:
            result += f"{count}: {effect}\n"
            count += 1
        return result
    
    def __repr__(self):
        return self.__str__()


class ParsingVariable:
    """
    Variable class for a better management during parsing\n
    A part in precondition or effect class.
    """
    def __init__(self):
        self.name = None

        # parameters of the variable, e.g. ['?a', '?b', '?i]
        # the type of the variable is decide by the action's parameters part
        self.parameters = []

    def __str__(self):
        return f"Variable({self.name}: {self.parameters})"
    
    def __repr__(self):
        return self.__str__()


class ParsingState:
    """
    State class for a better management during parsing\n
    A state is a part of the problem, which can be used to define the initial state of the world.
    """
    def __init__(self):
        self.variable = ParsingVariable()
        self.value: float | int | str = None
        self.target_variable = ParsingVariable()
    
    def __str__(self):
        return f"State(variable: {self.variable}, value: {self.value}, target_variable: {self.target_variable})"
    
    def __repr__(self):
        return self.__str__()


class ParsingCondition:
    """
    Condition class for a better management during parsing\n
    A precondition is a part of an action, which must be satisfied before the action can be executed.
    """
    def __init__(self):
        # logic operator, e.g. '=', '!=', '>', '<', '>=', '<='
        self.logic_operator = None

        self.state: ParsingState = ParsingState()
    def __str__(self):
        if self.state.value is not None:
            return f"Condition({self.state.variable} {self.logic_operator} {self.state.value})"
        return f"Condition({self.state.variable} {self.logic_operator} {self.state.target_variable})"
    
    def __repr__(self):
        return self.__str__()


class ParsingEpistemicCondition(ParsingCondition):
    """
    Epistemic condition class for a better management during parsing\n
    """
    def __init__(self):
        super().__init__()
        # epistemic logic operator, e.g. '=' or '!='
        self.epistemic_logic_operator = None

        # Belief sequence will show the belief relation from top to bottom
        # for example: [a,b,c] means B_aB_bB_c ...
        self.belief_sequence = []

        self.epistemic_truth = None
    
    def __str__(self):
        return f"EpistemicCondition(Belief sequence: {self.belief_sequence}, epistemic truth: \"{self.epistemic_logic_operator} {self.epistemic_truth}\", {super().__str__()})"

    def __repr__(self):
        return self.__str__()


class ParsingEffect:
    """
    Effect class for a better management during parsing\n
    An effect is a part of an action, which can be used to change the state of the world.
    """
    def __init__(self):
        # effect operator, e.g. 'increase', 'decrease', 'assign'
        self.effect_operator = None

        # variable that the effect is applied to
        self.variable = ParsingVariable()

        # a specific value that the effect is applied to, e.g. 1, 2, 3
        self.value = None

        # target variable, if the effect of the variable is not a number, then it will be a value comes from another variable, here called target variable
        self.target_variable = ParsingVariable()
    
    def __str__(self):
        if self.value is not None:
            return f"Effect('{self.effect_operator}' {self.variable} {self.value})"
        return f"Effect(effect_operator:'{self.effect_operator}' {self.variable} {self.target_variable})"
    
    def __repr__(self):
        return self.__str__()


class ParsingFunction:
    """
    Function class for a better management during parsing
    """
    def __init__(self):
        self.name: str = None

        # function parameters, e.g. {'agent': ['?a', '?b'], 'item': ['?i']}
        self.parameters: dict[str, list[str]] = dict()
    
    def __str__(self):
        return f"Function(name:{self.name}, parameters:{self.parameters}). "
    
    def __repr__(self):
        return self.__str__()


class ParsingRange:
    """
    Range class for a better management during parsing\n
    A range is a part of the problem, which can be used to define the range of a variable / function.
    """
    def __init__(self):
        self.function_name = None  # the variable that the range is applied to
        self.type = None  # the type of the variable, here, only 'integer', 'enumerate' are plausible
        self.enumerates: list[str] = None  # the enumerates of the function
        self.min: int = None # the minimum value of the range
        self.max: int = None # the maximum value of the range
    
    def __str__(self):
        if self.type == 'enumerate':
            return f"Range(function_name: {self.function_name}, type: {self.type}, enumerates: {self.enumerates})"
        else:
            return f"Range(function_name: {self.function_name}, type: {self.type}, min: {self.min}, max: {self.max})"
    
    def __repr__(self):
        return self.__str__()


class ParsingDomain:
    """
    Domain class for a better management during parsing
    """
    def __init__(self):
        self.name = None
        self.types: list[str] = []
        self.functions: list[ParsingFunction] = []
        self.actions: list[ParsingAction] = []

    def __str__(self):
        result = f"================= Domain \"({self.name})\" Parsing Result ================\n"
        result += f"Domain types: {self.types}\n"
        result += util.BIG_DIVIDER
        result += f"Domain functions:\n"
        for function in self.functions:
            result += f"{function}\n"
        result += util.BIG_DIVIDER
        result += f"Domain actions:\n"
        for action in self.actions:
            result += util.SMALL_DIVIDER
            result += f"{action}\n"
            
        return result
    
    def __repr__(self):
        return self.__str__()


class ParsingProblem:
    """
    Problem class for a better management during parsing
    """
    def __init__(self):
        self.domain_name = None
        self.problem_name = None
        self.agents = []
        self.objects: dict[str, list[str]] = dict()
        self.states: list[ParsingState] = dict()
        self.ranges: list[ParsingRange] = []
        self.goals: dict[str, list[ParsingCondition | ParsingEpistemicCondition]] = dict()
        self.acceptable_goal_set: list[str] = []
        self.max_belief_depth: int = 1
    
    def __str__(self):
        result = f"============ Problem \"({self.domain_name} : {self.problem_name})\" Parsing Result ===========\n"
        result += f"Agents: {self.agents}\n"
        result += util.BIG_DIVIDER
        result += f"Objects:\n"
        for object_name, object_list in self.objects.items():
            result += f"{object_name}: {object_list}\n"
        result += util.BIG_DIVIDER
        result += f"States:\n"
        for state in self.states:
            result += util.SMALL_DIVIDER
            result += f"{state}\n"
        result += util.BIG_DIVIDER
        result += f"Goals:\n"
        for agt_name, goal_list in self.goals.items():
            result += util.SMALL_DIVIDER
            result += f"{agt_name}:\n"
            for goal in goal_list:
                result += f"{goal}\n"
        result += util.BIG_DIVIDER
        result += f"Ranges:\n"
        for range in self.ranges:
            result += f"{range}\n"
        return result

    def __repr__(self):
        return self.__str__()


def convert_str_to_parsing_variable(variable_line: str, logger) -> ParsingVariable:
    variable = util.regex_search(SPLIT_VARIABLE_REGEX, variable_line, logger)
    var = ParsingVariable()
    var_name, var_params = variable[0]
    var.name = var_name
    var.parameters = var_params.split()
    return var


def convert_state_line_to_parsing_state(state_pair: tuple[str, str], logger) -> ParsingState:
    function_line, value = state_pair
    state = ParsingState()
    state.variable = convert_str_to_parsing_variable(function_line, logger)
    if util.regex_match(SPLIT_VARIABLE_REGEX, value):
        state.target_variable = convert_str_to_parsing_variable(value, logger)
    else:
        state.value = value
    return state


def convert_str_to_parsing_condition(condition_str: str, logger) -> ParsingCondition:
    # check whether the epistemic condition is present
    is_epistemic = '@ep' in condition_str
    epistemic_logic_operator = None
    belief_sequence = None
    logic_operator = None
    condition_variable = None
    condition_value = None
    epistemic_truth = None
    if is_epistemic:
        condition_str = util.regex_search(EPISTEMIC_CONDITION_SPLIT_REGEX, condition_str, logger)
        epistemic_logic_operator, belief_sequence, logic_operator, condition_variable, condition_value, epistemic_truth = condition_str[0]
    else:
        condition_str = util.regex_search(CONDITION_SPLIT_REGEX, condition_str, logger)
        logic_operator, condition_variable, condition_value = condition_str[0]
    
    if is_epistemic:
        precondition = ParsingEpistemicCondition()
    else:
        precondition = ParsingCondition()
    state = ParsingState()
    precondition.logic_operator = logic_operator
    state.variable = convert_str_to_parsing_variable(condition_variable, logger)

    # parse the condition value, if it is not a number, then it will parse the targetvariable
    if condition_value.isdigit():
        state.value = int(condition_value)
    elif ' ' not in condition_value:
        state.value = condition_value
    else:
        state.target_variable = convert_str_to_parsing_variable(condition_value, logger)
    precondition.state = state
    if is_epistemic:
        belief_sequence = util.regex_search(EXTRACT_BELIEF_AGENT_REGEX, belief_sequence, logger)
        for belief_agt in belief_sequence:
            precondition.belief_sequence.append(belief_agt)
            precondition.epistemic_logic_operator = epistemic_logic_operator
        precondition.condition = precondition
        precondition.epistemic_truth = epistemic_truth
    return precondition


class DomainParser:
    def __init__(self, handlers, log_level=DOMAIN_LOG_LEVEL):
        self.logger = util.setup_logger(__name__, handlers, logger_level=log_level)

    def run(self, file_path) -> ParsingDomain:
        self.logger.info(f"Domain \"{file_path}\" start initialization.")
        parsing_domain = ParsingDomain()

        if not os.path.isfile(file_path):
            self.logger.error(f"Domain Parser cannot find file \"{file_path}\"")
            exit(0)
        self.logger.info(f"Domain Parser found \"{file_path}\"")

        with open(file_path, 'r') as f:
            content = f.read()
            self.logger.info(f"Complete file reading.")

        try:
            # get domain name
            parsing_domain.name = self.get_name(content)
            self.logger.info(f"Domain name found")

            # get types
            parsing_domain.types = self.get_types(content)
            self.logger.info(f"Domain type found")

            # get functions
            parsing_domain.functions = self.get_functions(content)
            self.logger.info(f"Domain functions found")

            # get actions
            parsing_domain.actions = self.get_actions(content)
            self.logger.info(f"Domain actions found")
            self.logger.debug(f"Parsed Domain Result:\n{parsing_domain}")

            return parsing_domain
        except Exception as e:
            self.logger.error(f"An error occurred in parse domain stage")
            raise e
    
    def get_name(self, domain_content) -> str:
        ''' 
        Get domain name
        '''
        domain_name = util.regex_search(DOMAIN_NAME_REGEX, domain_content, self.logger)
        return domain_name[0]

    def get_types(self, domain_content) -> list[str]:
        '''
        Get domain types
        '''
        type_line = util.regex_search(TYPES_REGEX, domain_content, self.logger)
        type_line = type_line[0]
        return type_line.split()
    
    def get_functions(self, domain_content) -> list[ParsingFunction]:
        '''
        Get domain functions
        '''
        function_lines = util.regex_search(FUNCTION_EXTRACT_REGEX, domain_content, self.logger)
        function_lines = util.regex_search(FUNTION_SPLIT_NAME_PARAMETER_REGEX, function_lines[0], self.logger)
        functions = []
        for function_line in function_lines:
            function_name, parameter_part = function_line
            function = ParsingFunction()
            function.name = function_name
            parameter_part = util.regex_search(SPLIT_MULTI_PARAMETER_REGEX, parameter_part, self.logger)
            for parameters, parameter_type in parameter_part:
                function.parameters[parameter_type] = parameters.split()
            functions.append(function)
        return functions

    def get_actions(self, domain_content) -> list[ParsingAction]:
        '''
        Get actions scheme:\n
        <action name, parameters, preconditions, effects>\n
        recondition: logic operator ('=', '>', '<', '>=', '<='), variable, value or target variable\n
        effect: effect operator ('increase', 'decrease', 'assign'), variable, value or target variable\n
        variable: name, parameters
        '''
        action_lines = util.regex_search(ACTION_EXTRACT_REGEX, domain_content, self.logger)
        actions = []
        for action_name, parameter_part, precondition_part, effect_part in action_lines:
            action = ParsingAction()
            # parse action name
            action.name = action_name
            
            # parse parameters
            action.parameters = self.get_action_parameters(parameter_part)
            
            # parse the action preconditions
            action.pre_conditions = self.get_action_preconditions(precondition_part)
            
            # parse the action effects
            action.effects = self.get_action_effects(effect_part)
            
            actions.append(action)
        return actions
    
    def get_action_parameters(self, parameter_part: str) -> dict[str, list[str]]:
        """
        Get action parameters from the parameter part of an action\n
        :param parameter_part: the parameter part of an action
        :return: a dictionary with parameter type as key and a list of parameters as value
        """
        if not parameter_part: return {}
        parameter_part = util.regex_search(SPLIT_MULTI_PARAMETER_REGEX, parameter_part, self.logger)
        parameters = {}
        for parameters_str, parameter_type in parameter_part:
            parameters[parameter_type] = parameters_str.split()
        return parameters
    
    def get_action_preconditions(self, precondition_part: str) -> list[ParsingCondition]:
        """
        Get action preconditions from the precondition part of an action\n
        :param precondition_part: the precondition part of an action
        :return: a list of ParsingPrecondition objects
        """
        if not precondition_part: return []
        preconditions = []
        precondition_part = precondition_part.splitlines()
        for condition_part in precondition_part:
            preconditions.append(convert_str_to_parsing_condition(condition_part, self.logger))
        return preconditions
    
    def get_action_effects(self, effect_part: str) -> list[ParsingEffect]:
        """
        Get action effects from the effect part of an action\n
        :param effect_part: the effect part of an action
        :return: a list of ParsingEffect objects
        """
        if not effect_part: return []
        effect_part = util.regex_search(EFFECT_SPLIT_REGEX, effect_part, self.logger)
        effects = []
        for effect_operator, effect_variable, effect_value in effect_part:
            # create an effect
            effect = ParsingEffect()
            # set the effect operator
            effect.effect_operator = effect_operator
            
            # parse the effect variable
            variable = convert_str_to_parsing_variable(effect_variable, self.logger)
            effect.variable = variable
            
            # parse the effect value, if it is not a number, then it will parse the targetvariable
            if effect_value.isdigit():
                effect.value = int(effect_value)
            else:
                target_variable = convert_str_to_parsing_variable(effect_value, self.logger)
                effect.target_variable = target_variable
            
            effects.append(effect)
        return effects
            

class ProblemParser:
    """
    Problem parser class for parsing problem files in PDDL format.
    It will accept a folder path that contains distributed agent problem files.
    """
    def __init__(self, handlers, log_level=PROBLEM_LOG_LEVEL):
        self.logger = util.setup_logger(__name__, handlers, logger_level=log_level)
    
    def run(self, folder_path) -> ParsingProblem:
        self.logger.info(f"Problem \"{folder_path}\" start initialization.")
        parsing_problem = ParsingProblem()
        
        # get all files in the folder, split them into agent files and environment file
        agent_files, env_file = self.get_files(folder_path)
        log_agents = ""
        for agent_file in agent_files:
            log_agents += f"\n\"{agent_file}\""
        self.logger.info(f"Found {len(agent_files)} agent files:{log_agents}")
        self.logger.info(f"Found init file: \"{env_file}\"")

        # get environment content
        with open(env_file, 'r') as f:
            env_content = f.read()
            self.logger.info(f"Complete reading environment file.")

        # get agent contents
        agt_contents = []
        for agt_file in agent_files:
            with open(agt_file, 'r') as f:
                agt_contents.append(f.read())
                self.logger.info(f"Copmlete reading agent file \"{agt_file}\"")

        # check the validity of the problem contents
        if not self.check_validity(env_content, agt_contents):
            self.logger.error(f"Problem is not valid")
            raise Exception("Problem naming is not valid, may be one of the agtpddl has a wrong domain or problem name")
        
        try:
            # get names
            parsing_problem.domain_name, parsing_problem.problem_name = self.get_name(env_content)
            self.logger.info(f"Problem domain name \"{parsing_problem.domain_name}\" found\nProblem name \"{parsing_problem.problem_name}\" found")

            # get agents
            parsing_problem.agents = self.get_agents(env_content)
            self.logger.info(f"Problem agents found")

            # get objects
            parsing_problem.objects = self.get_objects(env_content)
            self.logger.info(f"Problem objects found")

            # get ranges
            parsing_problem.ranges = self.get_ranges(env_content)
            self.logger.info(f"Problem ranges found")
            
            # get initial states
            parsing_problem.states = self.get_init_states(parsing_problem.agents, env_content, agt_contents)
            self.logger.info(f"Problem initial states found")

            # get goals
            parsing_problem.goals = self.get_goals(parsing_problem.agents, agt_contents)
            self.logger.info(f"Problem goals found")

            # get acceptable goal set
            parsing_problem.acceptable_goal_set = self.get_acceptable_goal_set(env_content)
            self.logger.info(f"Problem acceptable goal set found")

            parsing_problem.max_belief_depth = self.get_max_belief_depth(env_content)
            self.logger.info(f"Problem max belief depth found")

            self.logger.debug(f"Parsed Problem Result:\n{parsing_problem}")
            return parsing_problem
        except Exception as e:
            self.logger.error(f"An error occurred in parse problem stage")
            raise e

        
    def get_files(self, folder_path):
        """
        Get all files in the folder, splite them into agent files and environment file.\n
        :param folder_path: the path of the folder
        :return: a tuple of (agent_files, environment_file)\n
        """
        if not os.path.isdir(folder_path):
            self.logger.error(f"Problem Parser cannot find folder \"{folder_path}\"")
            exit(0)

        # extract the files in the folder
        self.logger.info(f"Problem folder found \"{folder_path}\"")
        files = os.listdir(folder_path)
        files = list(map(lambda x: folder_path + '/' + x, files))
        agent_files = list(filter(lambda x: x.endswith('.agtpddl'), files))
        env_file = next(filter(lambda x: x.endswith('.envpddl'), files), None)
        if not agent_files or not env_file:
            self.logger.error(f"folder \"{folder_path}\" does not contains the required files.")
            exit(0)
        return agent_files, env_file
    
    def check_validity(self, env_content, agt_contents) -> bool:
        """
        Check the validity of the problem contents\n
        This will only check whether the name in each problem are valid.
        """
        # check the validity of the environment content
        problem_name, domain_name = self.get_name(env_content)
        for agt_content in agt_contents:
            agt_problem_name, agt_domain_name = self.get_name(agt_content)
            agt_name = util.regex_search(AGENT_NAME_REGEX, agt_content, self.logger)[0]


            if agt_problem_name != problem_name or agt_domain_name != domain_name:
                self.logger.error(f"The name of the problem in the agent file \"{agt_name}\" is not valid.\nProblem name: \"{problem_name}\", Domain name: \"{domain_name}\"")
                return False
        return True

    def get_name(self, env_content) -> tuple[str, str]:
        """
        Get domain name and problem name from the environment content
        """
        domain_name = util.regex_search(PROBLEM_DOMAIN_NAME_REGEX, env_content, self.logger)
        problem_name = util.regex_search(PROBLEM_NAME_REGEX, env_content, self.logger)
        return domain_name[0], problem_name[0]
        
    def get_agents(self, env_content) -> list[str]:
        """
        Get agents from the environment content
        """
        agents = util.regex_search(ENV_AGENTS_REGEX, env_content, self.logger)
        agents = agents[0].strip().split()
        return agents

    def get_objects(self, env_content) -> dict[str, list[str]]:
        """
        Get objects from the environment content
        """
        objects = dict()

        object_lines = util.regex_search(ENV_OBJECT_EXTRACT_REGEX, env_content, self.logger)
        object_lines = object_lines[0]
        if not object_lines:
            return objects
        object_lines = util.regex_search(ENV_OBJECT_SPLIT_REGEX, object_lines, self.logger)
        
        for this_objects, this_object_type in object_lines:
            this_objects = this_objects.split()
            objects[this_object_type] = this_objects
        return objects    

    def get_init_states(self, agents, env_content, agt_contents) -> dict[str, list[ParsingState]]:
        """
        Get initial states from the environment content
        """
        # set up the empty state set to each agents
        states: dict[str, list[ParsingState]] = dict()
        for agent in agents:
            states[agent] = []
        
        # 'unknown' here means the states in the world that no agents know.
        states = []
        
        # update unshared states
        unshared_state_pairs = self.parse_state_lines(INIT_STATE_EXTRACT_REGEX, env_content)
        for state_pairs in unshared_state_pairs:
            states.append(convert_state_line_to_parsing_state(state_pairs, self.logger))
        
        # update shared states
        # shared_state_pairs = self.parse_state_lines(SHARED_INIT_STATE_EXTRACT_REGEX, env_content)
        # for state_pair in shared_state_pairs:
        #     state = convert_state_line_to_parsing_state(state_pair, self.logger)
        #     for agent in agents:
        #         states[agent].append(state)
        
        # update individual states
        for agt_content in agt_contents:
            agt_name = util.regex_search(AGENT_NAME_REGEX, agt_content, self.logger)
            agt_name = agt_name[0]
        
        return states

    def get_ranges(self, env_content) -> list[ParsingRange]:
        """
        Get ranges from the problem content
        """
        ranges = []
        range_lines = util.regex_search(RANGES_EXTRACT_REGEX, env_content, self.logger)
        range_lines = range_lines[0]
        range_lines = util.regex_search(RANGES_SPLIT_REGEX, range_lines, self.logger)

        for function_name, type, range_values in range_lines:
            parsing_range = ParsingRange()
            parsing_range.function_name = function_name
            parsing_range.type = type
            if parsing_range.type == 'enumerate':
                parsing_range.range = range_values.split()
            elif parsing_range.type == 'integer':
                parsing_range.min, parsing_range.max = tuple(map(int, range_values.split(',')))
            elif parsing_range.type == 'float':
                parsing_range.min, parsing_range.max = tuple(map(float, range_values.split(',')))
            ranges.append(parsing_range)
        
        return ranges
    
    def parse_state_lines(self, regex, content) -> list[tuple[str, str]]:
        state_lines = util.regex_search(regex, content, self.logger)
        state_lines = state_lines[0]
        if not state_lines: return []
        state_lines = util.regex_search(INIT_STATE_SPLIT_REGEX, state_lines, self.logger)
        return state_lines

    def get_goals(self, agents, agt_contents) -> dict[str, list[ParsingCondition]]:
        goals = dict()
        for agent in agents:
            goals[agent] = []
        for agt_content in agt_contents:
            agt = util.regex_search(AGENT_NAME_REGEX, agt_content, self.logger)
            agt = agt[0]

            goal_lines = util.regex_search(GOAL_REGEX, agt_content, self.logger)
            goal_lines = goal_lines[0].splitlines()
            for goal_line in goal_lines:
                goals[agt].append(convert_str_to_parsing_condition(goal_line, self.logger))
        return goals

    def get_acceptable_goal_set(self, env_content) -> list[str]:
        goal_set_line = util.regex_search(GOAL_SET_REGEX, env_content, self.logger)
        goal_set = goal_set_line[0].split()
        # print(goal_set)
        return goal_set

    def get_max_belief_depth(self, env_content) -> int:
        max_belief_depth = util.regex_search(MAX_BELIEF_DEPTH_REGEX, env_content, self.logger)[0]
        return int(max_belief_depth)