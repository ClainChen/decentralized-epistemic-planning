import argparse
import logging
import sys
import traceback
import util
from epistemic_handler import model_builder, problem_builder

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

    options = parser.parse_args(sys.argv[1:])

    return options

if __name__ == '__main__':
    try:
        args = loadParameter()
        if args.c_logging_level:
            c_logging_level = LOGGING_LEVELS[args.c_logging_level]
        c_logging_display = args.c_logging_display
        handler = util.setup_logger_handlers('log/model_builder.log', log_mode='w',
                                             c_display=c_logging_display, c_logger_level=c_logging_level)
        logger = util.setup_logger(__name__, handlers=handler, logger_level=THIS_LOGGER_LEVEL)
        logger.info(f"Start building the model, type: \"{args.problem_type}\"")
        
        model = model_builder.build(args, handler)
        if not model.rules.check_model(model):
            logger.error(f"Model's functions are not following the rules.")
            print("Model's functions are not following the rules.")
            exit(0)
        logger.info(f"Model built successfully.")

        # problem_builder = problem_builder.ProblemBuilder(model, handler)
        # all_world = problem_builder.get_all_init_ontic_world()
        # agent_goal_set = problem_builder.get_all_poss_goals(depth=2)
        # problem_builder.get_all_poss_problem(all_world, agent_goal_set)
        # for agent, goals in agent_goal_set.items():
        #     result = f"\n{agent} goals:\n"
        #     for comb in goals:
        #         for g in comb:
        #             result += f"{g}\n"
        #         result += "\n"
        #     logger.debug(result)
        # print(sum([len(x) for x in agent_goal_set.values()]))
        # for world in all_world:
        #     result = "\n"
        #     for func in world:
        #         result += f"{func}\n"
        #     logger.debug(result)
        # print(len(all_world))


        model.simulate()

        print("Done.")
    except Exception as e:
        logger.error(f"{traceback.format_exc()}\n")
        print(f"{traceback.format_exc()}\n")
        print("Program failed caused by some reason. Please check the log file for more details.")

