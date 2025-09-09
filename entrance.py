import argparse
import logging
import sys
import traceback
import util
import time
from epistemic_handler import model_builder, problem_builder
import copy
import profile

c_logging_level = logging.INFO
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
    
    parser.add_argument('--cooperative', dest='problem_type', help='problem type controller\nwithout this key word will set the problem type to neutral', action='store_true')

    generate_problem_help = "add this argument will make the problem not to simulate\ninstead it will generate all possible problems based on the given domain and fundamental problem file"
    parser.add_argument('--generate_problem', dest='generate_problem', help=generate_problem_help, action='store_true')

    parser.add_argument('-tests', '--multi-tests', dest='num_multi_tests', type=int, help='The number of tests to run', default=1)

    parser.add_argument('-actions', '--action_sequence', dest='action_sequence_path', type=str.lower, help='The file of action sequence to run', default=None)
    options = parser.parse_args(sys.argv[1:])

    return options

if __name__ == '__main__':
    try:
        args = loadParameter()
        if args.c_logging_level:
            c_logging_level = LOGGING_LEVELS[args.c_logging_level]
        c_logging_display = args.c_logging_display
        log_name = f"{args.domain_path.split('/')[0].split('.')[0]}-{args.problem_path.split('/')[-1].split('.')[0]}-{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}.log"

        handler = util.setup_logger_handlers(f"log/{log_name}", log_mode='w',
                                             c_display=c_logging_display, c_logger_level=c_logging_level)
        logger = util.setup_logger(__name__, handlers=handler, logger_level=THIS_LOGGER_LEVEL)
        logger.info(f"Start building the model, type: \"{args.problem_type}\"")
        
        model = model_builder.build(args, handler)
        if not model.rules.check_model(model):
            logger.error(f"Model's functions are not following the rules.")
            print("Model's functions are not following the rules.")
            exit(0)
        logger.info(f"Model built successfully.")

        start_index = 0

        if args.action_sequence_path is not None:
            logger.info(f"Run under modified action sequence mode.")
            action_sequence = util.load_action_sequence(args.action_sequence_path, model, logger)
            for action in action_sequence:
                model.sim_move(action[0], action[1])
                # check whether the agents are complete their goals
            if model.full_goal_complete():
                print("Agent are complete their goals, simulate finish")
                exit(0)
            else:
                print("Agent didn't complete their goals, program will continue to simulate")
            start_index = model.get_agent_index_by_name(model.get_next_agent(action_sequence[-1][0]))
            # print each agent's current ep world
            # for agent in model.agents:
            #     ep_world = util.get_epistemic_world(model, [agent.name])
            #     output = ""
            #     for func in ep_world:
            #         output += f"{func}\n"
            #     logger.info(f"{agent.name} ep world:\n{output}")
            #     print(f"{agent.name}'s ep world:\n{output}")

        if util.check_bfs(model.copy()) == -1:
            logger.error(f"Model's goal setting do not have solution")
            print("Model's goal setting do not have solution")
            exit(0)

        if not args.generate_problem:
            for i in range(1, args.num_multi_tests + 1):
                print(f"{i}th Simulation:")
                running_model = copy.deepcopy(model)
                running_model.simulate(running_model.agents[start_index].name)
        else:
            problem_builder = problem_builder.ProblemBuilder(model, handler)
            problem_builder.generate_all_problem_pddl_files()

        print("Done.")
    except Exception as e:
        logger.error(f"{traceback.format_exc()}\n")
        print(f"{traceback.format_exc()}\n")
        print("Program failed caused by some reason. Please check the log file for more details.")

