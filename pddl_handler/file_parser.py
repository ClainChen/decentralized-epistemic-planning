import logging
import os
import re

import util
from .epistemic_class import BaseAction

DOMAIN_LOG_LEVEL = logging.DEBUG
DOMAIN_LOGGER_NAME = r"domain_parser"

# Domain related regexes
DOMAIN_NAME_REGEX = r"\(domain (\w+)\)"
TYPES_REGEX = r"\(:types$\n([\s\S]+?)\n^\s+\)"
FUNCTION_EXTRACT_REGEX = r"\(:functions$\n([\s\S]+?)^\s+\)"
FUNTION_SPLIT_NAME_PARAMETER_REGEX = r"\((\w+) ([^\)]+)\)"
SPLIT_MULTI_PARAMETER_REGEX = r"([^-\(]+)- (\w+)"
ACTION_EXTRACT_REGEX = r"\(:action (\w+)\n\s+:parameters (.+)\n\s+:precondition \(\n(\s+\([\s\S]+?)\s+\)\n\s+:effect \(\n(\s+\([\s\S]+?)\s+\)\n\s+\)"
PRECONDITION_SPLIT_REGEX = r"\(([!=><]=?|>|<) \(([^\)]+)\) \(?(\d|[^\)]+)\){1,2}"
SPLIT_VARIABLE_REGEX = r"(\w+) (.+)"
EFFECT_SPLIT_REGEX = r"\((increase|decrease|assign) \(([^\)]+)\) \(?(\d|[^\)]+)\){1,2}"

# Problem related regexes
PROBLEM_NAME_REGEX = r"\(problem (\w+)\)"
PROBLEM_DOMAIN_NAME_REGEX = r"\(:domain (\w+)\)"
AGENT_REGEX = r"\(:agents$\n([\s\S]+?)^\s+\)"
OBJECT_REGEX = r"\(:objects$\n([\s\S]+?)^\s+\)"
INIT_REGEX = r"\(:init$\n([\s\S]+?)^\s+\)"
GOAL_REGEX = r"\(:goal \(and$\n([\s\S]+?)^\s+\)\n\s+\)"
RANGES_REGEX = r"\(:ranges$\n([\s\S]+?)^\s+\)"


class ParsingAction:
    """
    Action class for a better management during parsing
    """
    def __init__(self):
        self.name = None
        self.parameters: dict[str, list[str]] = dict()
        self.precondition: list[ParsingPrecondition] = []
        self.effect: list[ParsingEffect] = []

    def __str__(self):
        result = f"Action:\n"
        result += f"Name: {self.name}\n"
        result += "Parameters:\n"
        for param_type, params in self.parameters.items():
            result += f"    {param_type} : {params}\n"
        result += "Preconditions:\n"
        count = 1
        for precondition in self.precondition:
            result += f"{count}: {precondition}\n"
            count += 1
        count = 1
        result += "Effects:\n"
        for effect in self.effect:
            result += f"{count}: {effect}\n"
            count += 1
        return result
    
    def __repr__(self):
        return self.__str__()


class ParsingPrecondition:
    """
    Precondition class for a better management during parsing\n
    A precondition is a part of an action, which must be satisfied before the action can be executed.
    """
    def __init__(self):
        # logic operator, e.g. '=', '>', '<', '>=', '<='
        self.logic_operator = None

        # variable that the precondition is applied to
        self.variable = ParsingVariable()

        # a specific value that the precondition is applied to, e.g. 1, 2, 3
        self.value = None

        # target variable, if the precondition of the variable is not a number, then it will be a value comes from another variable, here called target variable
        self.target_variable = ParsingVariable()

    def __str__(self):
        if self.value is not None:
            return f"Precondition({self.variable} {self.logic_operator} {self.value}). "
        return f"Precondition({self.variable} {self.logic_operator} {self.target_variable}). "
    
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
            return f"Effect('{self.effect_operator}' {self.variable} {self.value}). "
        return f"Effect(effect_operator:'{self.effect_operator}' {self.variable} {self.target_variable}). "
    
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


class DomainParser:
    def __init__(self, handlers, log_level=DOMAIN_LOG_LEVEL):
        self.logger = util.setup_logger(DOMAIN_LOGGER_NAME, handlers, logger_level=log_level)

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
            # self.logger.debug(f"Domain Parser content: \n{content}\n{util.LOGGER_DIVIDER}\n")

        try:
            # get domain name
            domain_name = util.regex_search(DOMAIN_NAME_REGEX, content, self.logger)
            parsing_domain.name = domain_name[0]
            self.logger.info(f"Domain name found")

            # get types
            type_line = util.regex_search(TYPES_REGEX, content, self.logger)
            type_line = type_line[0]
            parsing_domain.types = type_line.split()
            self.logger.info(f"Domain type found")

            # get functions
            function_lines = util.regex_search(FUNCTION_EXTRACT_REGEX, content, self.logger)
            function_lines = util.regex_search(FUNTION_SPLIT_NAME_PARAMETER_REGEX, function_lines[0], self.logger)
            for function_line in function_lines:
                function_name, parameter_part = function_line
                function = ParsingFunction()
                function.name = function_name
                parameter_part = util.regex_search(SPLIT_MULTI_PARAMETER_REGEX, parameter_part, self.logger)
                for parameters, parameter_type in parameter_part:
                    function.parameters[parameter_type] = parameters.split()
                parsing_domain.functions.append(function)
            self.logger.info(f"Domain functions found")

            '''
            get actions scheme:
            <action name, parameters, preconditions, effects>
            precondition: logic operator ('=', '>', '<', '>=', '<='), variable, value or target variable
            effect: effect operator ('increase', 'decrease', 'assign'), variable, value or target variable
            variable: name, parameters
            '''
            action_lines = util.regex_search(ACTION_EXTRACT_REGEX, content, self.logger)
            for action_name, parameter_part, precondition_part, effect_part in action_lines:
                action = ParsingAction()
                # parse action name
                action.name = action_name
                
                # parse parameters
                parameter_part = util.regex_search(SPLIT_MULTI_PARAMETER_REGEX, parameter_part, self.logger)
                for parameters, parameter_type in parameter_part:
                    action.parameters[parameter_type] = parameters.split()
                
                # parse the action preconditions
                precondition_part = util.regex_search(PRECONDITION_SPLIT_REGEX, precondition_part, self.logger)
                for condition_operator, condition_variable, condition_value in precondition_part:
                    # create a precondition
                    precondition = ParsingPrecondition()

                    # set the logic operator
                    precondition.logic_operator = condition_operator

                    # parse the condition variable
                    condition_variable = util.regex_search(SPLIT_VARIABLE_REGEX, condition_variable, self.logger)
                    variable = ParsingVariable()
                    variable_name, variable_params = condition_variable[0]
                    variable.name = variable_name
                    variable.parameters = variable_params.split()
                    precondition.variable = variable
                    
                    # parse the condition value, if it is not a number, then it will parse the target variable
                    if condition_value.isdigit():
                        precondition.value = int(condition_value)
                    else:
                        condition_value = util.regex_search(SPLIT_VARIABLE_REGEX, condition_value, self.logger)
                        target_varaible = ParsingVariable()
                        target_variable_name, target_variable_params = condition_value[0]
                        target_varaible.name = target_variable_name
                        target_varaible.parameters = target_variable_params.split()
                        precondition.target_variable = target_varaible
                    action.precondition.append(precondition)
                
                
                
                # parse the action effects
                effect_part = util.regex_search(EFFECT_SPLIT_REGEX, effect_part, self.logger)
                for effect_operator, effect_variable, effect_value in effect_part:
                    # create an effect
                    effect = ParsingEffect()

                    # set the effect operator
                    effect.effect_operator = effect_operator

                    # parse the effect variable
                    effect_variable = util.regex_search(SPLIT_VARIABLE_REGEX, effect_variable, self.logger)
                    variable = ParsingVariable()
                    variable_name, variable_params = effect_variable[0]
                    variable.name = variable_name
                    variable.parameters = variable_params.split()
                    effect.variable = variable
                    
                    # parse the effect value, if it is not a number, then it will parse the target variable
                    if effect_value.isdigit():
                        effect.value = int(effect_value)
                    else:
                        effect_value = util.regex_search(SPLIT_VARIABLE_REGEX, effect_value, self.logger)
                        target_varaible = ParsingVariable()
                        target_variable_name, target_variable_params = effect_value[0]
                        target_varaible.name = target_variable_name
                        target_varaible.parameters = target_variable_params.split()
                        effect.target_variable = target_varaible
                    action.effect.append(effect)
                
                parsing_domain.actions.append(action)
            self.logger.info(f"Domain actions found")
            self.logger.debug(f"Parsed Domain Result:\n{parsing_domain}")

            return parsing_domain
        except Exception as e:
            self.logger.error(f"An error occurred in parse domain stage: {e}")
            exit(0)


class ProblemParser:
    """
    Problem parser class for parsing problem files in PDDL format.
    It will accept a folder path that contains distributed agent problem files.
    """
    


