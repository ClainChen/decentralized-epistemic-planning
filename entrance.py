import argparse
import logging
import sys
import traceback

import util
from dep import model_builder
import json
import re
import copy

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

    fast_test_helper = "directly input the problem file path, the program will automatically parse all default parameters"
    parser.add_argument('-qt', '--quick-test', dest='quick_test', type=str.lower, help=fast_test_helper, default="")

    parser.add_argument('-d', '--domain', dest='domain_path', type=str.lower, help='domain file path')
    parser.add_argument('-p', '--problem', dest='problem_path', type=str.lower,
                        help='problem folder path\nplease make sure all of your distributed problem files are in the folder')

    parser.add_argument('-ob', '--observation-function', dest='observation_function', type=str.lower,
                        help='observation function file path\nthey are locate in observation_function folder\nthe name of them will be as same as the file name')

    parser.add_argument('--strategy', dest='strategy', type=str.lower,
                        help='The strategy you want to use\nthey are locate in policy_strategies folder\nthe name of them will be same as the file name',
                        default='share.py')

    parser.add_argument('--rules', dest='rules', type=str.lower,
                        help='rules file path\nthe rules are locate in rules folder\nthe name of rules will be same as the file name.')

    debug_mode_help = ('set the console logging level, the strength ordered by:\n'
                       'debug > info > warning > error > critical')

    parser.add_argument('--log-level', dest='c_logging_level', type=str.lower, help=debug_mode_help, default='info')
    parser.add_argument('--log-display', dest='c_logging_display', action='store_true',
                        help='add this argument will display the full log in the console')

    parser.add_argument('--share', dest='problem_type',
                        help='problem type controller\nwithout this key word will set the problem type to unknown goal settings',
                        action='store_true')

    generate_problem_help = "add this argument will make the problem not to simulate\ninstead it will generate all possible problems based on the given domain and fundamental problem file"
    parser.add_argument('--generate_problem', dest='generate_problem', help=generate_problem_help, action='store_true')

    parser.add_argument('-tests', '--multi_tests', dest='num_multi_tests', type=int, help='The number of tests to run',
                        default=1)

    parser.add_argument('-actions', '--action_sequence', dest='action_sequence_path', type=str.lower,
                        help='The file of action sequence to run', default=None)

    parser.add_argument('--multi_strategies', dest="multi_strategies", type=flexible_dict_type,
                        help='Config for agents has different strategies', default={})

    parser.add_argument('--multi_ob', dest="multi_observation_functions", type=flexible_dict_type,
                        help='Config for agents has different observation functions', default={})

    parser.add_argument('--without_agt_goal', dest='without_agt_goal', type=list,
                        help='The given agents will not use the goal filter', default=[])

    parser.add_argument('--without_agt_exp', dest='without_agt_exp', type=list,
                        help='The given agents will not update their experience of actions', default=[])

    parser.add_argument('-goals', dest='num_goals', type=int,
                        help='The maximum number of goals for each agent to generate problems', default=2)

    options = parser.parse_args(sys.argv[1:])

    return options


if __name__ == '__main__':
    try:
        args = loadParameter()
        if args.quick_test:
            domain_name = args.quick_test.split('/')[0]
            args.problem_path = args.quick_test
            args.domain_path = f"{domain_name}/domain.pddl"
            if not args.observation_function:
                args.observation_function = f"{domain_name}.py"
            if not args.rules:
                args.rules = f"{domain_name}.py"
            if not args.problem_type:
                args.problem_type = True
        if args.strategy == 'shareexp.py':
            args.consider_exp = True
        else:
            args.consider_exp = False

        if args.c_logging_level:
            c_logging_level = LOGGING_LEVELS[args.c_logging_level]
        c_logging_display = args.c_logging_display
        pp = args.problem_path.split('/')
        log_strategy = f"{pp[1]}.log"

        handler = util.setup_logger_handlers(f"log/{pp[0]}/{log_strategy}", log_mode='w',
                                             c_display=c_logging_display, c_logger_level=c_logging_level)
        util.LOGGER = util.setup_logger(__name__, handlers=handler, logger_level=THIS_LOGGER_LEVEL)
        # util.LOGGER.info(f"Start building the model, type: \"{args.problem_type}\"")

        util.LIMIT = args.num_goals

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
            # if model.full_goal_complete():
            #     print("Agent are complete their goals, simulate finish")
            #     exit(0)
            # else:
            #     print("Agent didn't complete their goals, program will continue to simulate")
            # for f in model.ontic_functions:
            #     print(f)
            # for f in util.OBS_FUNC['a'].get_observable_functions(model, model.ontic_functions, 'a'):
            bs = ['a']
            print("-------------------")
            print(f'ep of {bs}:')
            for f in util.get_epistemic_world(model, bs):
                print(f)
            # for act in model.get_agent_successors('b'):
            #     print(act.header())
            # print(model.get_agent_by_name('b').print_poss_goals())
            exit(0)
            start_index = model.get_agent_index_by_name(model.get_next_agent(action_sequence[-1][0]))
        print("-------------------")
        path_len, path = util.check_bfs(model.copy())
        # exit(0)
        if path_len == -1:
            util.LOGGER.error(f"Model's goal setting do not have solution")
            print("Model's goal setting do not have solution")
            exit(0)

        print(f"Path Length: {path_len}")
        print(f"Path: {path}")
        print("-------------------")

        step_lst = []
        time_lst = []
        vms = []
        call_of_jp = []
        for i in range(1, args.num_multi_tests + 1):
            print(f"{i}th Simulation:")
            running_model = model.duplicate()
            steps, time_used, num_vms = running_model.simulate(running_model.agents[start_index].name)
            step_lst.append(steps)
            time_lst.append(time_used)
            vms.append(num_vms)
            cojp = util.CALL_OF_JP
            call_of_jp.append(cojp)
            util.CALL_OF_JP = 0
        util.LOGGER.exp(f"Avg Steps: {sum(step_lst) / len(step_lst)}\n"
                        f"Avg Time: {(sum(time_lst) / len(time_lst)):.6f}s\n"
                        f"Avg VMS: {sum(vms) / len(vms)}\n")
        print(f"Avg Steps: {sum(step_lst) / len(step_lst)}\n"
              f"Avg Time: {(sum(time_lst) / len(time_lst)):.6f}s\n"
              f"Avg VMS: {sum(vms) / len(vms)}\n"
              f"Avg Call of JP: {sum(call_of_jp) / len(call_of_jp)}\n")
    except Exception as e:
        util.LOGGER.error(f"{traceback.format_exc()}\n")
        print(f"{traceback.format_exc()}\n")
        print("Program failed caused by some reason. Please check the log file for more details.")
