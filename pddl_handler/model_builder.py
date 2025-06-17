import argparse
import logging
import sys

import util
from . import file_parser as fp

logging_levels = {'critical': logging.CRITICAL,
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
    args = loadParameter()
    c_logging_level = logging.INFO
    if args.c_logging_level:
        c_logging_level = logging_levels[args.c_logging_level]
    c_logging_display = args.c_logging_display
    handler = util.setup_logger_handlers('log/model_builder.log', log_mode='w',
                                         c_display=c_logging_display, c_logger_level=c_logging_level)
    domain_parser = fp.DomainParser(handler)
    domain_path = util.MODEL_FOLDER_PATH + args.domain_path
    domain_parser.run(domain_path)
    

