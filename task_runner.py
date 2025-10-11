import concurrent.futures
import subprocess
import argparse
import shlex
from pathlib import Path
import json

# 要执行的命令列表
commands = [
# # coin1
#     "python entrance.py -d coin/domain.pddl -p coin/problem1 -ob coin.py --rules coin.py --strategy experiment/share.py --share -test 100",

#     "python entrance.py -d coin/domain.pddl -p coin/problem1 -ob coin.py --rules coin.py --strategy experiment/stay.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d coin/domain.pddl -p coin/problem1 -ob coin.py --rules coin.py --strategy experiment/coop.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d coin/domain.pddl -p coin/problem1 -ob coin.py --rules coin.py --strategy experiment/coopexp.py --without_agt_goal [a,b] -test 100",

#     "python entrance.py -d coin/domain.pddl -p coin/problem1 -ob coin.py --rules coin.py --strategy experiment/filtergoal.py --without_agt_exp [a,b] -test 100",
    
#     "python entrance.py -d coin/domain.pddl -p coin/problem1 -ob coin.py --rules coin.py --strategy experiment/filtergoalexp.py -test 100",

# # coin2
#     "python entrance.py -d coin/domain.pddl -p coin/problem2 -ob coin.py --rules coin.py --strategy experiment/share.py --share -test 100",

#     "python entrance.py -d coin/domain.pddl -p coin/problem2 -ob coin.py ---rules coin.py -strategy experiment/stay.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d coin/domain.pddl -p coin/problem2 -ob coin.py --rules coin.py --strategy experiment/coop.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d coin/domain.pddl -p coin/problem2 -ob coin.py --rules coin.py --strategy experiment/coopexp.py --without_agt_goal [a,b] -test 100",

#     "python entrance.py -d coin/domain.pddl -p coin/problem2 -ob coin.py --rules coin.py --strategy experiment/filtergoal.py --without_agt_exp [a,b] -test 100",
    
#     "python entrance.py -d coin/domain.pddl -p coin/problem2 -ob coin.py --rules coin.py --strategy experiment/filtergoalexp.py -test 100",

# # corridor2a1i
#     "python entrance.py -d corridor/domain.pddl -p corridor/2a1i_1 -ob corridor.py --rules corridor.py --strategy experiment/share.py --share -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a1i_1 -ob corridor.py --rules corridor.py --strategy experiment/stay.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a1i_1 -ob corridor.py --rules corridor.py --strategy experiment/coop.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a1i_1 -ob corridor.py --rules corridor.py --strategy experiment/coopexp.py --without_agt_goal [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a1i_1 -ob corridor.py --rules corridor.py --strategy experiment/filtergoal.py --without_agt_exp [a,b] -test 100",
    
#     "python entrance.py -d corridor/domain.pddl -p corridor/2a1i_1 -ob corridor.py --rules corridor.py --strategy experiment/filtergoalexp.py -test 100",

# # corridor2a2i_1
#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_1 -ob corridor.py --rules corridor.py --strategy experiment/share.py --share -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_1 -ob corridor.py --rules corridor.py --strategy experiment/stay.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_1 -ob corridor.py --rules corridor.py --strategy experiment/coop.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_1 -ob corridor.py --rules corridor.py --strategy experiment/coopexp.py --without_agt_goal [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_1 -ob corridor.py --rules corridor.py --strategy experiment/filtergoal.py --without_agt_exp [a,b] -test 100",
    
#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_1 -ob corridor.py --rules corridor.py --strategy experiment/filtergoalexp.py -test 100",

# # corridor2a2i_2
#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_2 -ob corridor.py --rules corridor.py --strategy experiment/share.py --share -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_2 -ob corridor.py --rules corridor.py --strategy experiment/stay.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_2 -ob corridor.py --rules corridor.py --strategy experiment/coop.py --without_agt_goal [a,b] --without_agt_exp [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_2 -ob corridor.py --rules corridor.py --strategy experiment/coopexp.py --without_agt_goal [a,b] -test 100",

#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_2 -ob corridor.py --rules corridor.py --strategy experiment/filtergoal.py --without_agt_exp [a,b] -test 100",
    
#     "python entrance.py -d corridor/domain.pddl -p corridor/2a2i_2 -ob corridor.py --rules corridor.py --strategy experiment/filtergoalexp.py -test 100",

# corridor4a4i_1
    "python entrance.py -d corridor/domain.pddl -p corridor/4a4i_1 -ob corridor.py --rules corridor.py --strategy experiment/share.py --share -test 10",

    "python entrance.py -d corridor/domain.pddl -p corridor/4a4i_1 -ob corridor.py --rules corridor.py --strategy experiment/stay.py --without_agt_goal [a,b,c,d] --without_agt_exp [a,b,c,d] -test 10",

    "python entrance.py -d corridor/domain.pddl -p corridor/4a4i_1 -ob corridor.py --rules corridor.py --strategy experiment/coop.py --without_agt_goal [a,b,c,d] --without_agt_exp [a,b,c,d] -test 10",

    "python entrance.py -d corridor/domain.pddl -p corridor/4a4i_1 -ob corridor.py --rules corridor.py --strategy experiment/coopexp.py --without_agt_goal [a,b,c,d] -test 10",

    # "python entrance.py -d corridor/domain.pddl -p corridor/4a4i_1 -ob corridor.py --rules corridor.py --strategy experiment/filtergoal.py --without_agt_exp [a,b,c,d] -test 100",
    
    # "python entrance.py -d corridor/domain.pddl -p corridor/4a4i_1 -ob corridor.py --rules corridor.py --strategy experiment/filtergoalexp.py -test 100",

# grapevine4a1s1d
    "python entrance.py -d grapevine/domain.pddl -p grapevine/4a1s1d -ob grapevine.py --rules grapevine.py --strategy experiment/share.py --share -test 10",

    "python entrance.py -d grapevine/domain.pddl -p grapevine/4a1s1d -ob grapevine.py --rules grapevine.py --strategy experiment/stay.py --without_agt_goal [a,b,c,d] --without_agt_exp [a,b,c,d] -test 10",

    "python entrance.py -d grapevine/domain.pddl -p grapevine/4a1s1d -ob grapevine.py --rules grapevine.py --strategy experiment/coop.py --without_agt_goal [a,b,c,d] --without_agt_exp [a,b,c,d] -test 10",

    "python entrance.py -d grapevine/domain.pddl -p grapevine/4a1s1d -ob grapevine.py --rules grapevine.py --strategy experiment/coopexp.py --without_agt_goal [a,b,c,d] -test 10",

    "python entrance.py -d grapevine/domain.pddl -p grapevine/4a1s1d -ob grapevine.py --rules grapevine.py --strategy experiment/filtergoal.py --without_agt_exp [a,b,c,d] -test 10",
    
    "python entrance.py -d grapevine/domain.pddl -p grapevine/4a1s1d -ob grapevine.py --rules grapevine.py --strategy experiment/filtergoalexp.py -test 10",
]


def dict_to_pretty_str(data_dict, indent=2):
    """
    使用json格式美化字典输出
    """
    return json.dumps(data_dict, indent=indent, ensure_ascii=False)

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

def parse_command_line(command_line):
    """
    解析命令行参数并返回包含指定信息的字典
    
    Args:
        command_line: 完整的命令行字符串
        
    Returns:
        dict: 包含 -d, -p, -ob, --strategy, --rules 信息的字典
    """
    # 分割命令行参数
    if isinstance(command_line, str):
        args = shlex.split(command_line)
    else:
        args = command_line
    
    # 构建结果字典
    result = {
        'domain': args[3],
        'problem': args[5],
        'ob': args[7],
        'rules': args[9],
        'strategy': args[11]
    }
    
    return result

def run_command(command):
    command += " --log-level experiment --log-display"
    parse = parse_command_line(command)
    """执行单条命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=3600)
        result = {
            'command': command,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        return parse, result
    except subprocess.TimeoutExpired:
        result = {
            'command': command,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Command timed out'
        }
        return parse, result
    except Exception as e:
        result = {
            'command': command,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }
        return parse, result

# 使用线程池并行执行
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # 提交所有任务
    future_to_command = {executor.submit(run_command, cmd): cmd for cmd in commands}
    
    # 获取结果
    for future in concurrent.futures.as_completed(future_to_command):
        command = future_to_command[future]
        try:
            parse, result = future.result()
            dire = Path(f'out/experiments/{parse['problem']}')
            filename = f'{parse['strategy'][11:-3]}.txt'
            output = f"命令: {result['command']}\n"
            output += f"返回码: {result['returncode']}\n"
            if result['stdout']:
                output += f"输出: {result['stdout']}\n"
            if result['stderr']:
                output += f"错误: {result['stderr']}"
            
            if not dire.exists():
                dire.mkdir(parents=True)
            with open(dire / filename, 'w') as f:
                f.write(output)

            task = f"{parse['problem'].replace('/', '-')}-{parse['strategy'][11:-3]}"

            print(f"任务 {task} 执行完成")
        except Exception as exc:
            print(f"任务 {command} 执行时发生异常: {exc}")