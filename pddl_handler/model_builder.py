import argparse
import logging
import sys
import traceback

import util
from .file_parser import *
from .epistemic_class import *

THIS_LOGGER_LEVEL = logging.DEBUG
LOGGING_LEVELS = {'critical': logging.CRITICAL,
                  'fatal': logging.FATAL,
                  'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'warn': logging.WARN,
                  'info': logging.INFO,
                  'debug': logging.DEBUG,
                  'notset': logging.NOTSET}

class CustomHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


def loadParameter():
    parser = argparse.ArgumentParser(description='HELP of model-builder parser', formatter_class=CustomHelpFormatter)

    parser.add_argument('-d', '--domain', dest='domain_path', help='domain file path', required=True)
    parser.add_argument('-p', '--problem', dest='problem_path', help='problem folder path, please make sure all of your distributed problem files are in the folder', required=True)

    debug_mode_help = ('set the console logging level, the strength ordered by:\n'
                       'debug > info > warning > error > critical')

    parser.add_argument('--log-level', dest='c_logging_level', type=str.lower, help=debug_mode_help, default='info')
    parser.add_argument('--log-display', dest='c_logging_display', action='store_true',
                        help='add this argument will display the full log in the console')
    
    parser.add_argument('-t', '--type', dest='problem_type', type=str.lower, help='The type of problem, only allows COOPERATIVE or NATURAL', choices=['cooperative', 'natural'], default='cooperative')

    options = parser.parse_args(sys.argv[1:])

    return options


def main():
    try:
        args = loadParameter()
        c_logging_level = logging.INFO
        if args.c_logging_level:
            c_logging_level = LOGGING_LEVELS[args.c_logging_level]
        c_logging_display = args.c_logging_display
        handler = util.setup_logger_handlers('log/model_builder.log', log_mode='w',
                                             c_display=c_logging_display, c_logger_level=c_logging_level)
        logger = util.setup_logger(__name__, handlers=handler, logger_level=THIS_LOGGER_LEVEL)
        logger.info(f"Start building the model, type: \"{args.problem_type}\"")

        domain_parser = DomainParser(handler)
        domain_path = util.MODEL_FOLDER_PATH + args.domain_path
        domain: ParsingDomain = domain_parser.run(domain_path)

        problem_parser = ProblemParser(handler)
        problem_path = util.MODEL_FOLDER_PATH + args.problem_path
        problem: ParsingProblem = problem_parser.run(problem_path)

        checker = ModelChecker(domain, problem, handler)
        check_result = checker.check_validity()
        if not check_result:
            logger.error(f"Model is invalid.")
            raise f"Model did not pass the checker."
        
        model = build(domain, problem, args.problem_type, logger)
    except Exception as e:
        logger.error(f"{traceback.format_exc()}\n")
        print(f"Model building failed.")
        raise e


def build(domain: ParsingDomain, problem: ParsingProblem, problem_type: str, logger) -> Model:
    model = Model()
    model.domain_name = domain.name
    model.problem_name = problem.problem_name

    # build entities
    for agent in problem.agents:
        new_entity = Entity()
        new_entity.name = agent
        new_entity.type = 'agent'
        model.entities.append(new_entity)
    for type, objects in problem.objects.items():
        for object in objects:
            new_entity = Entity()
            new_entity.name = object
            new_entity.type = type
            model.entities.append(new_entity)
    
    # build base states
    for funtion in domain.functions:
        new_state = BaseState()
        new_state.name = funtion.name
        new_state.require_entities = copy.deepcopy(funtion.parameters)
        model.base_states.append(new_state)
    
    # build actions
    for action in domain.actions:
        new_action = BaseAction()
        new_action.name = action.name
        new_action.parameters = copy.deepcopy(action.parameters)
        # build preconditions
        for condition in action.pre_condition:
            new_condition = parsing_condition_to_condition_unit(condition, new_action.parameters)
            new_action.pre_conditions.append(new_condition)
        # build effects
        for effect in action.effect:
            new_effect = EffectUnit()
            new_effect.operator = EFFECT_OPERATOR[effect.effect_operator]
            new_effect.value = effect.value
            new_effect.variable = parsing_variable_to_base_state(effect.variable, new_action.parameters)
            new_effect.target_variable = parsing_variable_to_base_state(effect.target_variable, new_action.parameters)
            new_action.effects.append(new_effect)
        model.actions.append(new_action)
    
    for action in model.actions:
        logger.debug(f"{action}")
    
    # build ranges
    for range in problem.ranges:
        new_range = Range()
        new_range.name = range.function_name
        new_range.type = RANGE_TPYE[range.type]
        new_range.enumerates = range.enumerates
        new_range.min = range.min
        new_range.max = range.max
        model.ranges.append(new_range)
    
    # build agents
    for agent in problem.agents:
        new_agent = Agent()
        new_agent.name = agent
        # build states
        for state in problem.states[agent]:
            new_state = State()
            new_state.name = state.variable.name
            new_state.value = state.value
            for param in state.variable.parameters:
                for entity in model.entities:
                    if param == entity.name:
                        new_state.entities.append(entity)
            new_agent.states.append(new_state)
        # build goals
        for goal in problem.goals[agent]:
            new_goal = parsing_condition_to_condition_unit(goal)
            new_agent.goals.append(new_goal)
        model.agents.append(new_agent)
    
    logger.info(f"Model built")
    logger.debug(f"Model:\n{model}")

    return model

            
def parsing_variable_to_base_state(variable: ParsingVariable, parameters=None) -> BaseState:
    new_state = BaseState()
    new_state.name = variable.name
    params = variable.parameters
    if parameters:
        for param in params:
            for type, params in parameters.items():
                if param in params:
                    new_state.require_entities[type].append(param)
    else:
        new_state.entities = copy.deepcopy(params)
    return new_state


def parsing_condition_to_condition_unit(condition: ParsingCondition, parameters=None) -> ConditionUnit:
    new_condition = ConditionUnit()
    copying_condition = copy.deepcopy(condition)
    if isinstance(condition, ParsingEpistemicCondition):
        new_condition = EpistemicConditionUnit()
        new_condition.epistemic_logic_operator = EPISTEMIC_TRUTH_OPERATOR[copying_condition.epistemic_logic_operator]
        new_condition.belief_sequence = copying_condition.belief_sequence
        new_condition.epistemic_truth = EPISTEMIC_TRUTH[copying_condition.epistemic_truth]
    new_condition.operator = QUANTITY_OPERATOR[copying_condition.logic_operator]
    new_condition.variable = parsing_variable_to_base_state(copying_condition.state.variable, parameters)
    new_condition.value = copying_condition.state.value
    new_condition.target_variable.name = parsing_variable_to_base_state(copying_condition.state.target_variable, parameters)
    return new_condition