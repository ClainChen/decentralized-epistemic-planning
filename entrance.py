import argparse
import logging
import sys
import traceback
import util
from epistemic_handler import model_builder

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

    parser.add_argument('-d', '--domain', dest='domain_path', help='domain file path', required=True)
    parser.add_argument('-p', '--problem', dest='problem_path', help='problem folder path, please make sure all of your distributed problem files are in the folder', required=True)
    parser.add_argument('-ob', '--observation-function', dest='observation_function_path', help='observation function file path', required=True)
    parser.add_argument('--strategy', dest='strategy', type=str.lower, help='The strategy you want to use, the strategies are locate in policy_strategies folder, the name of strategy will be same as the file name', default='random.py')

    debug_mode_help = ('set the console logging level, the strength ordered by:\n'
                       'debug > info > warning > error > critical')

    parser.add_argument('--log-level', dest='c_logging_level', type=str.lower, help=debug_mode_help, default='info')
    parser.add_argument('--log-display', dest='c_logging_display', action='store_true',
                        help='add this argument will display the full log in the console')
    
    parser.add_argument('--cooperative', dest='problem_type', help='problem type controller, without this key word will set the problem type to neutral', action='store_true')

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
        logger.info(f"Model built successfully.")
        all_functions = model.generate_all_possible_functions()
        model.simulate()

        print("Done.")
    except Exception as e:
        logger.error(f"{traceback.format_exc()}\n")
        print("Model building failed.")
        print("Program failed caused by some reason. Please check the log file for more details.")

