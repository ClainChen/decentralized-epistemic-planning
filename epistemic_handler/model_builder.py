import util
from epistemic_handler.file_parser import *
from epistemic_handler.epistemic_class import *
from epistemic_handler.model_checker import *

LOGGER_LEVEL = logging.DEBUG

def build(args, handler) -> Model:
    try:
        logger = util.setup_logger(__name__, handlers=handler, logger_level=LOGGER_LEVEL)
        logger.info(f"Start building the model, type: \"{args.problem_type}\"")

        domain, problem = parse_file(args, handler, logger)
        checker = ModelChecker(domain, problem, handler)
        if not checker.check_validity():
            logger.error("Model is invalid")
            exit(1)
        model = build_model(domain, problem, handler, logger, args)
        return model
    except Exception as e:
        raise e

def parse_file(args, handler, logger):
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
        raise Exception("Model did not pass the checker.")
    
    return domain, problem


def build_model(domain: ParsingDomain, problem: ParsingProblem, handler, logger, args):
    try:
        model = Model()
        model.init(handler, 
                "cooperative" if args.problem_type else "neutral", 
                util.OBS_FUNC_FOLER_PATH + args.observation_function, 
                util.STRATEGY_FOLDER_PATH + args.strategy,
                util.RULES_FOLDER_PATH + args.rules)
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
        for parsing_state in problem.states:
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
            parsing_goals = problem.goals[agent]
            for parsing_goal in parsing_goals:
                
                new_goal = Condition()
                if isinstance(parsing_goal, ParsingEpistemicCondition):
                    new_goal.ep_operator = EPISTEMIC_OPERATOR_MAPS[parsing_goal.epistemic_logic_operator]
                    new_goal.belief_sequence = util.remove_continue_duplicates(parsing_goal.belief_sequence)
                    new_goal.ep_truth = EPISTEMIC_TRUTH_MAPS[parsing_goal.epistemic_truth]
                new_goal.condition_operator = CONDITION_OPERATOR_MAPS[parsing_goal.logic_operator]
                condition_function_schema = model.get_function_schema_by_name(parsing_goal.state.variable.name)
                new_goal.condition_function_name = parsing_goal.state.variable.name
                new_goal.condition_function_parameters = dict(zip(condition_function_schema.require_parameters.keys(), parsing_goal.state.variable.parameters))
                if not parsing_goal.state.value is None:
                    new_goal.value = parsing_goal.state.value
                else:
                    target_function_schema = model.get_function_schema_by_name(parsing_goal.state.target_variable.name)
                    new_goal.target_function_name = parsing_goal.state.target_variable.name
                    new_goal.target_function_parameters = dict(zip(target_function_schema.require_parameters.keys(), parsing_goal.state.target_variable.parameters))
                new_agent.own_goals.append(new_goal)
            model.agents.append(new_agent)

        if model.problem_type == ProblemType.COOPERATIVE:
            for agent1 in model.agents:
                for agent2 in model.agents:
                    if agent1.name != agent2.name:
                        agent1.other_goals[agent2.name] = agent2.own_goals

        logger.debug(f"Model:\n{model}")
        # if not check_goal_conflicts(model):
        #     logger.error("Goal conflicts found")
        #     raise Exception("Goal conflicts found")
        # logger.info(f"Pass goal conflict check")
        return model
    except Exception as e:
        print("Model building failed.")
        logger.error("Model building failed.")
        raise e


    