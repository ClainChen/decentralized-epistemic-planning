import argparse
import logging
import sys
import traceback
import util
import time
from epistemic_handler import model_builder, problem_builder
import copy
import json
import profile
import re



c_logging_level = logging.INFO
THIS_LOGGER_LEVEL = 25
LOGGING_LEVELS = {'critical': logging.CRITICAL,
                  'fatal': logging.FATAL,
                  'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'warn': logging.WARN,
                  'experiment': 25,
                  'info': logging.INFO,
                  'debug': logging.DEBUG,
                  'notset': logging.NOTSET}

class CustomHelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

def flexible_dict_type(value):
    
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        fixed_value = value.replace("'", '"')
        fixed_value = re.sub(r'([\{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_value)
        fixed_value = re.sub(r':\s*([a-zA-Z0-9/_-]*\.py)\s*([,\}])', r':"\1"\2', fixed_value)
        return json.loads(fixed_value)


def loadParameter():
    parser = argparse.ArgumentParser(description='HELP of model-builder parser', formatter_class=CustomHelpFormatter)

    parser.add_argument('-d', '--domain', dest='domain_path', type=str.lower, help='domain file path', required=True)
    parser.add_argument('-p', '--problem', dest='problem_path', type=str.lower, help='problem folder path\nplease make sure all of your distributed problem files are in the folder', required=True)

    parser.add_argument('-ob', '--observation-function', dest='observation_function', type=str.lower, help='observation function file path\nthey are locate in observation_function folder\nthe name of them will be as same as the file name', required=True)

    parser.add_argument('--strategy', dest='strategy', type=str.lower, help='The strategy you want to use\nthey are locate in policy_strategies folder\nthe name of them will be same as the file name', default='random.py')

    parser.add_argument('--rules', dest='rules', type=str.lower, help='rules file path\nthe rules are locate in rules folder\nthe name of rules will be same as the file name.', required=True)

    debug_mode_help = ('set the console logging level, the strength ordered by:\n'
                       'debug > info > warning > error > critical')

    parser.add_argument('--log-level', dest='c_logging_level', type=str.lower, help=debug_mode_help, default='info')
    parser.add_argument('--log-display', dest='c_logging_display', action='store_true',
                        help='add this argument will display the full log in the console')
    
    parser.add_argument('--share', dest='problem_type', help='problem type controller\nwithout this key word will set the problem type to unknown goal settings', action='store_true')

    generate_problem_help = "add this argument will make the problem not to simulate\ninstead it will generate all possible problems based on the given domain and fundamental problem file"
    parser.add_argument('--generate_problem', dest='generate_problem', help=generate_problem_help, action='store_true')

    parser.add_argument('-tests', '--multi_tests', dest='num_multi_tests', type=int, help='The number of tests to run', default=1)

    parser.add_argument('-actions', '--action_sequence', dest='action_sequence_path', type=str.lower, help='The file of action sequence to run', default=None)

    parser.add_argument('--multi_strategies', dest="multi_strategies", type=flexible_dict_type, help='Config for agents has different strategies', default={})

    parser.add_argument('--multi_ob', dest="multi_observation_functions", type=flexible_dict_type, help='Config for agents has different observation functions', default={})

    parser.add_argument('--without_agt_goal', dest='without_agt_goal', type=list, help='The given agents will not use the goal filter', default=[])

    parser.add_argument('--without_agt_exp', dest='without_agt_exp', type=list, help='The given agents will not update their experience of actions', default=[])

    options = parser.parse_args(sys.argv[1:])

    return options

if __name__ == '__main__':
    try:
        args = loadParameter()
        if args.c_logging_level:
            c_logging_level = LOGGING_LEVELS[args.c_logging_level]
        c_logging_display = args.c_logging_display
        log_name = f"{args.problem_path.replace('/', '-')}-{args.strategy[11:-3]}.log"
        
        handler = util.setup_logger_handlers(f"log/{log_name}", log_mode='w',
                                             c_display=c_logging_display, c_logger_level=c_logging_level)
        util.LOGGER = util.setup_logger(__name__, handlers=handler, logger_level=THIS_LOGGER_LEVEL)
        util.LOGGER.info(f"Start building the model, type: \"{args.problem_type}\"")
        
        model = model_builder.build(args)
        # t.diagnose_model_serialization(model)
        

        if not util.RULES.check_model(model):
            util.LOGGER.error(f"Model's functions are not following the rules.")
            print("Model's functions are not following the rules.")
            exit(0)
        util.LOGGER.info(f"Model built successfully.")

        start_index = 0

        if args.action_sequence_path is not None:
            util.LOGGER.info(f"Run under modified action sequence mode.")
            action_sequence = util.load_action_sequence(args.action_sequence_path, model)
            for action in action_sequence:
                model.sim_move(action[0], action[1])
                # check whether the agents are complete their goals
            if model.full_goal_complete():
                print("Agent are complete their goals, simulate finish")
                exit(0)
            else:
                print("Agent didn't complete their goals, program will continue to simulate")
            for f in model.ontic_functions:
                print(f)
            start_index = model.get_agent_index_by_name(model.get_next_agent(action_sequence[-1][0]))

        # path_len = util.check_bfs(model.copy())
        # if path_len == -1:
        #     util.LOGGER.error(f"Model's goal setting do not have solution")
        #     print("Model's goal setting do not have solution")
        #     exit(0)

        # print(f"Standard Path Length: {path_len}")

        if not args.generate_problem:
            step_lst = []
            time_lst = []
            for i in range(1, args.num_multi_tests + 1):
                print(f"{i}th Simulation:")
                running_model = copy.deepcopy(model)
                steps, time_used = running_model.simulate(running_model.agents[start_index].name)
                step_lst.append(steps)
                time_lst.append(time_used)
            util.LOGGER.exp(f"Avg Steps: {sum(step_lst) / len(step_lst)}\nAvg Time: {(sum(time_lst) / len(time_lst)):.6f}s")
        else:
            problem_builder = problem_builder.ProblemBuilder(model)
            problem_builder.generate_all_problem_pddl_files()

        print("Done.")
    except Exception as e:
        util.LOGGER.error(f"{traceback.format_exc()}\n")
        print(f"{traceback.format_exc()}\n")
        print("Program failed caused by some reason. Please check the log file for more details.")

