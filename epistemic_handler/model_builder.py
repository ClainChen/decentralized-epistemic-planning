import util
from epistemic_handler.file_parser import *
from epistemic_handler.epistemic_class import *
from epistemic_handler.model_checker import *
from epistemic_handler.problem_builder import *
from itertools import permutations
import pickle
from pathlib import Path

LOGGER_LEVEL = logging.DEBUG

def build(args) -> Model:
    try:
        util.LOGGER.info(f"Start building the model, type: \"{args.problem_type}\"")

        domain, problem = parse_file(args)
        checker = ModelChecker(domain, problem)
        if not checker.check_validity():
            util.LOGGER.error("Model is invalid")
            exit(1)
        model = build_model(domain, problem, args)
        return model
    except Exception as e:
        raise e

def parse_file(args):
    domain_parser = DomainParser()
    domain_path = util.MODEL_FOLDER_PATH + args.domain_path
    domain: ParsingDomain = domain_parser.run(domain_path)

    problem_parser = ProblemParser()
    problem_path = util.MODEL_FOLDER_PATH + args.problem_path
    problem: ParsingProblem = problem_parser.run(problem_path)

    checker = ModelChecker(domain, problem)
    check_result = checker.check_validity()
    if not check_result:
        util.LOGGER.error(f"Model is invalid.")
        raise Exception("Model did not pass the checker.")
    
    return domain, problem


def build_model(domain: ParsingDomain, problem: ParsingProblem, args):
    try:
        util.load_observation_function(util.OBS_FUNC_FOLER_PATH + args.observation_function)
        util.load_rules(util.RULES_FOLDER_PATH + args.rules)
        util.load_policy_strategy(util.STRATEGY_FOLDER_PATH + args.strategy)

        model = Model()
        model.init("cooperative" if args.problem_type else "neutral")
        model.domain_name = domain.name
        model.problem_name = problem.problem_name
        model.max_belief_depth = problem.max_belief_depth

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
        for r in problem.ranges:
            name = r.function_name
            if r.type == 'integer':
                ranges[name] = (r.min, r.max)
            else:
                ranges[name] = r.enumerates
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
        
        # build all poss functions
        id = 0
        header_id = 0
        for func_schema in model.function_schemas:
            params_group = [model.get_all_entity_name_by_type(type) for type in func_schema.require_parameters.values()]
            params = [list(e) for e in itertools.product(*params_group)]
            for param in params:
                parameters = dict(zip(func_schema.require_parameters.keys(), param))
                header_id += 1
                if func_schema.type == ValueType.INTEGER:
                    minn, maxx = func_schema.range
                    for v in range(minn, maxx + 1):
                        id += 1
                        func = Function()
                        func.id = id
                        func.header_id = header_id
                        func.name = func_schema.name
                        func.parameters = parameters
                        func.value = v
                        model.ALL_FUNCS.add_function(func)
                else:
                    for v in func_schema.range:
                        id += 1
                        func = Function()
                        func.id = id
                        func.header_id = header_id
                        func.name = func_schema.name
                        func.parameters = parameters
                        func.value = v
                        model.ALL_FUNCS.add_function(func)
        
        # print(model.ALL_FUNCS)
        # exit(0)
        
        # build the goal set S_G
        for ag in problem.acceptable_goal_set:
            sg = Condition()
            schema = model.get_function_schema_by_name(ag.function_name)
            
            sg.condition_function_name = ag.function_name
            sg.condition_function_parameters = dict(zip(schema.require_parameters.keys(), ag.parameters))
            values = []
            if ag.values[0] == "all":
                if schema.type == ValueType.INTEGER:
                    values = list(range(schema.range[0], schema.range[1] + 1))
                elif schema.type == ValueType.ENUMERATE:
                    values = schema.range
            elif schema.type == ValueType.INTEGER:
                values = list(range(ag.values[0], ag.values[1] + 1))
            else:
                values = ag.values
            for v in values:
                sgp = Condition()
                sgp.condition_function_name = sg.condition_function_name
                sgp.condition_function_parameters = sg.condition_function_parameters
                sgp.condition_operator = ConditionOperator.EQUAL
                sgp.value = v
                model.S_G.add(sgp)
        
        # build initial functions
        # this includes all initial functions that are not in epistemic world
        for parsing_state in problem.states:
            function_schema = model.get_function_schema_by_name(parsing_state.variable.name)
            name = function_schema.name
            params = dict(zip(function_schema.require_parameters.keys(), parsing_state.variable.parameters))
            value = parsing_state.value
            # print(name, params, value)
            func = model.ALL_FUNCS.get_function(name, params, value)
            model.ontic_functions.append(func)
        
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
                new_effect_schema.target_function_belief_sequence = effect.target_variable_belief_sequence
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
        
        # build possible belief sequences
        agents = [agent.name for agent in model.agents]
        belief_sequences = []
        for d in range(1, model.max_belief_depth + 1):
            belief_sequences += [list(comb) for comb in list(permutations(agents, d))]
        model.possible_belief_sequences = belief_sequences

        if model.problem_type == ProblemType.NEUTRAL:
            # Here add a pickle process to store all possible goals for specific problem setup
            # So the next time, when loading the same problem, it will directly load the goals from the pickle file
            # But please make sure you will not change the problem setup inside the problem file, you can just create a new problem, copy it or whatever, change the name of problem, and do what u want to do.
            has_backup = True
            folder_path = Path(f"possible_goals_backup/{model.domain_name}-{model.problem_name}")
            if not folder_path.exists():
                folder_path.mkdir(parents=True)
                has_backup = False
            for agent in model.agents:
                if folder_path / f"{agent.name}.pkl" not in folder_path.iterdir():
                    has_backup = False
                    break
            if not has_backup:
                hint = "No goal backup found or the backup is not complete, now start to build the possible goals.\n"
                hint += f"The result will automatically be stored in the folder: possible_goals_backup/{model.domain_name}-{model.problem_name}.\n"
                hint += f"Please do not change the problem setup inside the problem file, \nyou can just create a new problem, copy it or whatever, \nhange the name of problem, and do what you want to do.\n"
                util.LOGGER.info(hint)
                print(hint)
                problemBuilder = ProblemBuilder(model)
                for agent in model.agents:
                    all_goals, avg_time = problemBuilder.get_all_poss_goals(agent.name)
                    agent.max_time = avg_time + 10
                    agent.all_possible_goals = [cell for cell in all_goals if set(cell[agent.name]) == set(agent.own_goals)]                    
                    with open(folder_path / f"{agent.name}.pkl", "wb") as f:
                        pickle.dump(agent.all_possible_goals, f)
            else:
                hint = f"The goal backup found in the folder: possible_goals_backup/{model.domain_name}-{model.problem_name}, now loading the goals from the backup."
                util.LOGGER.info(hint)
                print(hint)
                for agent in model.agents:
                    with open(folder_path / f"{agent.name}.pkl", "rb") as f:
                        agent.all_possible_goals = pickle.load(f)

        util.LOGGER.debug(f"Model:\n{model}")
        # if not check_goal_conflicts(model):
        #     util.LOGGER.error("Goal conflicts found")
        #     raise Exception("Goal conflicts found")
        # util.LOGGER.info(f"Pass goal conflict check")
        return model
    except Exception as e:
        print("Model building failed.")
        util.LOGGER.error("Model building failed.")
        raise e


    