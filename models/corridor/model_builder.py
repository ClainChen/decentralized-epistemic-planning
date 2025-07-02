import argparse
import logging
import sys
import traceback

import util
from pddl_handler.file_parser import *
from pddl_handler.epistemic_class import *
from .problem_builder import *

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
    parser.add_argument('-ob', '--observation-function', dest='observation_function_path', help='observation function file path', required=True)

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
        
        model = build(domain, problem, args.problem_type, args.observation_function_path, logger, handler)
        model.run_a_round()

    except Exception as e:
        logger.error(f"{traceback.format_exc()}\n")
        print(f"Model building failed.")
        raise e


def build(domain: ParsingDomain, problem: ParsingProblem, problem_type: str, observation_function_path: str, logger, handler) -> Model:
    model = Model(handler, util.MODEL_FOLDER_PATH + observation_function_path)
    model.domain_name = domain.name
    model.problem_name = problem.problem_name

    # build entities
    for agent_name in problem.agents:
        new_entity = Entity(agent_name, 'agent')
        model.entities.append(new_entity)
    for type, objects in problem.objects.items():
        for object in objects:
            new_entity = Entity(object, type)
            model.entities.append(new_entity)

    # build ranges of each function schemas
    ranges = {}
    for range in problem.ranges:
        name = range.function_name
        if range.type == 'integer':
            ranges[name] = (range.min, range.max)
        else:
            ranges[name] = range.enumerates
    
    # build function schemas
    for parsing_function in domain.functions:
        function_schema = FunctionSchema()
        function_schema.name = parsing_function.name
        function_schema.range = ranges[parsing_function.name]
        if isinstance(function_schema.range, tuple):
            function_schema.type = ValueType.INTEGER
        else:
            function_schema.type = ValueType.ENUMERATE
        for type, parameters in parsing_function.parameters.items():
            for parameter in parameters:
                function_schema.require_parameters[parameter] = type
        model.function_schemas.append(function_schema)
    
    # build initial functions
    # this includes all initial functions that are not in epistemic world
    for parsing_states in problem.states.values():
        for parsing_state in parsing_states:
            function_schema = model.get_function_schema_by_name(parsing_state.variable.name)
            new_function = Function()
            new_function.name = function_schema.name
            new_function.range = function_schema.range
            new_function.type = function_schema.type
            new_function.value = parsing_state.value
            new_function.parameters = dict(zip(function_schema.require_parameters.keys(), parsing_state.variable.parameters))
            model.ontic_functions.append(new_function)
    
    # build action schemas
    for parsing_action in domain.actions:
        new_action_schema = ActionSchema()
        new_action_schema.name = parsing_action.name
        for type, parameters in parsing_action.parameters.items():
            for parameter in parameters:
                new_action_schema.require_parameters[parameter] = type
        
        # build action schema's pre condition schemas
        for parsing_condition in parsing_action.pre_conditions:
            new_condition_schema = ConditionSchema()
            if isinstance(parsing_condition, ParsingEpistemicCondition):
                new_condition_schema.belief_sequence = parsing_condition.belief_sequence
                new_condition_schema.ep_operator = EPISTEMIC_OPERATOR_MAPS[parsing_condition.epistemic_logic_operator]
                new_condition_schema.ep_truth = EPISTEMIC_TRUTH_MAPS[parsing_condition.epistemic_truth]
            new_condition_schema.condition_operator = CONDITION_OPERATOR_MAPS[parsing_condition.logic_operator]
            new_condition_schema.condition_function_schema = model.get_function_schema_by_name(parsing_condition.state.variable.name)

            util.swap_param_orders(new_condition_schema.condition_function_schema, parsing_condition.state.variable)

            if parsing_condition.state.value is not None:
                new_condition_schema.value = parsing_condition.state.value
            else:
                new_condition_schema.target_function_schema = model.get_function_schema_by_name(parsing_condition.state.target_variable.name)
                util.swap_param_orders(new_condition_schema.target_function_schema, parsing_condition.state.target_variable)
            new_action_schema.pre_condition_schemas.append(new_condition_schema)
            
        # build effects
        for effect in parsing_action.effects:
            new_effect_schema = EffectSchema()
            new_effect_schema.effect_type = EFFECT_TYPE_MAPS[effect.effect_operator]
            new_effect_schema.effect_function_schema = model.get_function_schema_by_name(effect.variable.name)
            util.swap_param_orders(new_effect_schema.effect_function_schema, effect.variable)
            if effect.value is not None:
                new_effect_schema.value = effect.value
            else:
                new_effect_schema.target_function_schema = model.get_function_schema_by_name(effect.target_variable.name)
                util.swap_param_orders(new_effect_schema.target_function_schema, effect.target_variable)
            new_action_schema.effect_schemas.append(new_effect_schema)
        model.action_schemas.append(new_action_schema)
        
    # build agents
    for agent in problem.agents:
        new_agent = Agent()
        new_agent.name = agent
        parsing_states = problem.states[agent]
        for parsing_state in parsing_states:
            function_schema = model.get_function_schema_by_name(parsing_state.variable.name)
            new_function = Function()
            new_function.name = function_schema.name
            new_function.range = function_schema.range
            new_function.type = function_schema.type
            new_function.value = parsing_state.value
            new_function.parameters = dict(zip(function_schema.require_parameters.keys(), parsing_state.variable.parameters))
            new_agent.functions.append(new_function)
        parsing_goals = problem.goals[agent]
        for parsing_goal in parsing_goals:
            new_goal = Goal()
            if isinstance(parsing_goal, ParsingEpistemicCondition):
                new_goal.ep_operator = EPISTEMIC_OPERATOR_MAPS[parsing_goal.epistemic_logic_operator]
                new_goal.belief_sequence = parsing_goal.belief_sequence
                new_goal.ep_truth = EPISTEMIC_TRUTH_MAPS[parsing_goal.epistemic_truth]
            new_goal.condition_operator = CONDITION_OPERATOR_MAPS[parsing_goal.logic_operator]
            new_goal.goal_function_name = parsing_goal.state.variable.name
            new_goal.goal_function_parameters = parsing_goal.state.variable.parameters
            if parsing_goal.state.value is not None:
                new_goal.value = parsing_goal.state.value
            else:
                new_goal.target_function_name = parsing_goal.state.target_variable.name
                new_goal.target_function_parameters = parsing_goal.state.target_variable.parameters
            new_agent.Goals.append(new_goal)
        model.agents.append(new_agent)

    logger.debug(f"Model:\n{model}")
    return model






