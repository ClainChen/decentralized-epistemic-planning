import argparse
import logging
import sys
import traceback

import util
from . import file_parser

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

        domain_parser = file_parser.DomainParser(handler)
        domain_path = util.MODEL_FOLDER_PATH + args.domain_path
        domain: file_parser.ParsingDomain = domain_parser.run(domain_path)

        problem_parser = file_parser.ProblemParser(handler)
        problem_path = util.MODEL_FOLDER_PATH + args.problem_path
        problem: file_parser.ParsingProblem = problem_parser.run(problem_path)

        checker = file_parser.ModelChecker(domain, problem, handler)
        check_result = checker.check_validity()
        if not check_result:
            logger.error(f"Model is invalid.")
            raise f"Model did not pass the checker."
    except Exception as e:
        logger.error(f"{traceback.format_exc()}\n")
        print(f"Model building failed.")
        raise e

