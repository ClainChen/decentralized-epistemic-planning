import concurrent.futures
import subprocess
import argparse
import shlex
from pathlib import Path
import json



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

def analyze_tasks(tasks):
    for task in tasks:
        domain: str = task['model_name']
        problems: list[dict] = task['problems']
        ob: str = task['ob']
        rules: str = task['rules']
        tests: list[dict] = task['tests']
        for problem in problems:
            if problem['enabled'] == 0:
                continue
            agents = f"{problem['agents']}".replace("\'", '')
            agents = agents.replace(" ", '')
            for test in tests:
                if test['enabled'] == 0:
                    continue
                strategy: str = test['strategy']
                num_tests: int = test['num']
                if strategy == 'share':
                    command = f"python entrance.py -d {domain}/domain.pddl -p {domain}/{problem['name']} -ob {ob} --rules {rules} --strategy experiment/{strategy}.py --share -tests {num_tests}"
                elif strategy == 'stay':
                    command = f"python entrance.py -d {domain}/domain.pddl -p {domain}/{problem['name']} -ob {ob} --rules {rules} --strategy experiment/{strategy}.py --without_agt_goal '{agents}' --without_agt_exp '{agents}' -tests {num_tests}"
                elif strategy == 'stayexp':
                    command = f"python entrance.py -d {domain}/domain.pddl -p {domain}/{problem['name']} -ob {ob} --rules {rules} --strategy experiment/{strategy}.py --without_agt_goal '{agents}' -tests {num_tests}"
                elif strategy == 'coop':
                    command = f"python entrance.py -d {domain}/domain.pddl -p {domain}/{problem['name']} -ob {ob} --rules {rules} --strategy experiment/{strategy}.py --without_agt_goal '{agents}' --without_agt_exp '{agents}' -tests {num_tests}"
                elif strategy == 'coopexp':
                    command = f"python entrance.py -d {domain}/domain.pddl -p {domain}/{problem['name']} -ob {ob} --rules {rules} --strategy experiment/{strategy}.py --without_agt_goal '{agents}' -tests {num_tests}"
                elif strategy == 'filtergoal':
                    command = f"python entrance.py -d {domain}/domain.pddl -p {domain}/{problem['name']} -ob {ob} --rules {rules} --strategy experiment/{strategy}.py --without_agt_exp '{agents}' -tests {num_tests} -goals {problem['limit']}"
                else:
                    command = f"python entrance.py -d {domain}/domain.pddl -p {domain}/{problem['name']} -ob {ob} --rules {rules} --strategy experiment/{strategy}.py -tests {num_tests} -goals {problem['limit']}"
                yield command

with open('tasks.json', 'r') as f:
    tasks = json.load(f)

# 使用线程池并行执行
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    # 提交所有任务
    future_to_command = {executor.submit(run_command, cmd): cmd for cmd in analyze_tasks(tasks)}
    
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